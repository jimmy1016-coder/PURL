import math
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import transformers
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from data.data_utils import (
    write_pairs_output_txt,
)
from metrics import calculate_eer_and_min_dcf
from models import (
    Encoder,
    Projector,
)

class BaseTrainer(pl.LightningModule):
    def __init__(
        self,
        encoder: Encoder,
        projector: Projector,
        loss_func: nn.Module,
        learning_rate: float,
        lr_scheduler_type: str,
        sample_rate: int,
        optim_weight_decay: Optional[float] = None,
        samples_per_epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
        clustering_model: Optional[KMeans] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "encoder",
                "projector",
                "loss_func",
            ]
        )

        self.encoder = encoder
        self.projector = projector
        self.clustering_model = clustering_model
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.optim_weight_decay = optim_weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.sample_rate = sample_rate

        # params needed for lr scheduler, not a very clean solution but works
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size

    def _get_loaded_ckpt_epoch(self):
        checkpoint = torch.load(self.trainer.ckpt_path)
        return checkpoint["epoch"]

    def forward(self, x):
        return self.encoder(x)

    def test_step(self, batch, batch_idx):
        sig, path = batch
        emb = self.encoder(sig.squeeze(1))

        for p, e in zip(path, emb):
            self.test_emb_dict[p] = e.squeeze().to("cpu").numpy()

    def on_test_epoch_start(self):
        self.test_emb_dict = {}
        self.test_labels = self.trainer.test_dataloaders.dataset.labels

        self.test_checkpoint_epoch = self._get_loaded_ckpt_epoch()

    def on_test_epoch_end(self):
        output_dir_path = Path(self.logger.save_dir) / "evaluation_results"
        output_dir_path.mkdir(parents=True, exist_ok=True)

        files1 = self.trainer.test_dataloaders.dataset.files1
        files2 = self.trainer.test_dataloaders.dataset.files2

        eer, min_dcf = calculate_eer_and_min_dcf(
            emb_dict=self.test_emb_dict, files1=files1, files2=files2, labels=self.test_labels
        )

        output_txt_file = output_dir_path / (
            Path(self.trainer.test_dataloaders.dataset.pairs_txt_file).stem
            + f"_scores_epoch{self.test_checkpoint_epoch}.txt"
        )
        write_pairs_output_txt(
            self.test_emb_dict,
            files1,
            files2,
            output_txt_file,
        )

        torch.save(self.encoder.model.state_dict(), output_dir_path / "model.pt")

        eer_message = f"EER = {eer * 100:.2f} %\n"
        mindcf_message = f"minDCF @ p=0.05 = {min_dcf:.4f}\n"
        full_output_message = eer_message + mindcf_message

        metrics_txt_file = output_dir_path / (
            Path(self.trainer.test_dataloaders.dataset.pairs_txt_file).stem
            + f"_metrics_epoch{self.test_checkpoint_epoch}.txt"
        )

        with open(metrics_txt_file, "w") as file:
            file.write(full_output_message)

        print(full_output_message)
        print(f"Output scores and metrics saved to {output_dir_path}.")

    def on_predict_start(self):
        if self.clustering_model is None:
            raise RuntimeError("Clustering model is not provided in config.")
        loaded_ckpt_epoch = self._get_loaded_ckpt_epoch()
        self.cluster_output_file = Path(self.logger.save_dir) / f"clusters_k{self.clustering_model.n_clusters}_epoch{loaded_ckpt_epoch}.pkl"
        print("COMPUTING EMBEDDING CENTROIDS")

    def predict_step(self, batch):
        spec, speaker_id = batch

        embs = self.encoder(spec.squeeze())
        embs = self.projector(embs)

        centroid = embs.mean(dim=0).squeeze()

        return str(speaker_id[0]), centroid.detach().cpu().numpy()

    def cluster(self, embedding_centroids: np.ndarray):
        print("RUNNING CLUSTERING")

        embedding_centroids = normalize(embedding_centroids)
        cluster_labels = self.clustering_model.fit_predict(embedding_centroids)

        return cluster_labels

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.optim_weight_decay,
        )
        if self.lr_scheduler_type == "cosine":
            assert (
                self.samples_per_epoch and self.batch_size
            ), "Set samples_per_epoch and batch_size trainer arguments when using cosine scheduler!"
            print("USING COSINE LR")
            steps_per_epoch = math.ceil(self.samples_per_epoch / self.batch_size / self.trainer.accumulate_grad_batches)
            num_total_steps = steps_per_epoch * self.trainer.max_epochs
            num_warmup_steps = int(0.1 * num_total_steps)

            lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                opt,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_total_steps,
            )

            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate_schedule",
            }
            return {
                "optimizer": opt,
                "lr_scheduler": lr_scheduler_config,
            }
        elif self.lr_scheduler_type == "constant":
            return opt
        else:
            ValueError("Only 'cosine' and 'constant' schedulers are supported.")


class SupervisedTrainer(BaseTrainer):
    def validation_step(self, batch, batch_idx):
        sig, path = batch
        emb = self.encoder(sig.squeeze(1))

        for p, e in zip(path, emb):
            self.valid_emb_dict[p] = e.squeeze().to("cpu").numpy()

    def on_validation_epoch_start(self):
        self.files1 = self.trainer.val_dataloaders.dataset.files1
        self.files2 = self.trainer.val_dataloaders.dataset.files2
        self.valid_labels = self.trainer.val_dataloaders.dataset.labels
        self.valid_emb_dict = {}

    def on_validation_epoch_end(self):
        eer, min_dcf = calculate_eer_and_min_dcf(
            emb_dict=self.valid_emb_dict, files1=self.files1, files2=self.files2, labels=self.valid_labels
        )

        self.log(name="valid/eer", value=eer)
        self.log(
            name="valid/min_dcf",
            value=min_dcf,
        )


class SSLContrastiveTrainer(BaseTrainer):
    def _log_audio(self, audios, prefix):
        audio1 = audios[0, 0]
        audio2 = audios[0, 1]

        self.logger.experiment.add_audio(
            f"{prefix}/audio1",
            audio1.to("cpu"),
            self.global_step,
            self.sample_rate,
        )
        self.logger.experiment.add_audio(
            f"{prefix}/audio2",
            audio2.to("cpu"),
            self.global_step,
            self.sample_rate,
        )

    def _step(self, prefix, batch):
        features, audios = batch

        if self.global_step < 5 and self.global_rank == 0:
            self._log_audio(audios, prefix)

        features1 = features[:, 0]
        features2 = features[:, 1]

        # input shape: [batch, time, features]
        embs = self.encoder(
            torch.cat(
                [features1, features2],
                dim=0,
            )
        )
        embs = self.projector(embs)
        # output shape: [batch, 1, emb_size]

        emb1, emb2 = torch.split(embs, features.shape[0])

        loss, metrics = self.loss_func(
            emb1.squeeze(1),
            emb2.squeeze(1),
        )

        on_step = True if prefix == "train" else False

        for k, v in metrics.items():
            self.log(
                f"{prefix}/{k}",
                float(v),
                on_step=on_step,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch),
            )

        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("valid", batch)


class SupConTrainer(SupervisedTrainer):
    def _log_audio(self, audios, prefix):
        audio1 = audios[0, 0]
        audio2 = audios[0, 1]

        self.logger.experiment.add_audio(
            f"{prefix}/audio1",
            audio1.to("cpu"),
            self.global_step,
            self.sample_rate,
        )
        self.logger.experiment.add_audio(
            f"{prefix}/audio2",
            audio2.to("cpu"),
            self.global_step,
            self.sample_rate,
        )

    def training_step(self, batch):
        features, audios, labels = batch

        if self.global_step < 5 and self.global_rank == 0:
            self._log_audio(audios, "train")

        features1 = features[:, 0]
        features2 = features[:, 1]

        # input shape: [batch, time, features]
        embs = self.encoder(
            torch.cat(
                [features1, features2],
                dim=0,
            )
        )
        embs = self.projector(embs)
        # output shape: [batch, 1, emb_size]

        emb1, emb2 = torch.split(embs, features.shape[0])

        loss, metrics = self.loss_func(
            emb1.squeeze(1),
            emb2.squeeze(1),
            labels,
        )

        for k, v in metrics.items():
            self.log(
                f"train/{k}",
                float(v),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=len(batch),
            )

        return loss


class ClassificationTrainer(SupervisedTrainer):
    def __init__(
        self,
        encoder: Encoder,
        projector: Projector,
        clustering_model: KMeans,
        loss_func: nn.Module,
        learning_rate: float,
        optim_weight_decay: float,
        lr_scheduler_type: str,
        sample_rate: int,
        samples_per_epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        super().__init__(
            encoder=encoder,
            projector=projector,
            clustering_model=clustering_model,
            loss_func=loss_func,
            learning_rate=learning_rate,
            optim_weight_decay=optim_weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            sample_rate=sample_rate,
            samples_per_epoch=samples_per_epoch,
            batch_size=batch_size,
        )
        self.train_metrics = self._build_metrics_collection("train")

    def _build_metrics_collection(self, prefix):
        metrics = MetricCollection(
            {
                f"{prefix}/accuracy": Accuracy(task="multiclass", num_classes=self.projector.num_classes),
                f"{prefix}/precision": Precision(task="multiclass", num_classes=self.projector.num_classes),
                f"{prefix}/recall": Recall(task="multiclass", num_classes=self.projector.num_classes),
                f"{prefix}/macrof1": F1Score(
                    task="multiclass", average="macro", num_classes=self.projector.num_classes
                ),
            }
        )

        return metrics

    def _log_audio(self, audio, prefix):
        self.logger.experiment.add_audio(
            f"{prefix}/audio",
            audio.to("cpu"),
            self.global_step,
            self.sample_rate,
        )

    def _log_metrics(self, predictions, targets, loss, prefix):
        self.train_metrics(
            torch.argmax(predictions, dim=-1),
            torch.argmax(targets, dim=-1),
        )

        on_step = True if prefix == "train" else False

        self.log(f"{prefix}/loss", loss, on_step=on_step, on_epoch=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=False)

    def _step(self, prefix, batch):
        features, audios, labels = batch

        if self.global_step < 5 and self.global_rank == 0:
            self._log_audio(audios[0], prefix)

        # input shape: [batch, time, features]
        embs = self.encoder(features)
        logits = self.projector(embs)  # in this case the projector is the classifier
        # output shape: [batch, 1, emb_size]

        if isinstance(self.loss_func, nn.CrossEntropyLoss):
            loss = self.loss_func(logits, labels.argmax(dim=-1, keepdim=False))
        else:
            loss = self.loss_func(logits, labels.argmax(dim=-1, keepdim=True))

        predictions = torch.softmax(logits.squeeze(), dim=1)
        targets = labels.int().squeeze()

        self._log_metrics(predictions=predictions, targets=targets, loss=loss.mean(), prefix=prefix)

        return loss

    def training_step(self, batch):
        return self._step("train", batch)
