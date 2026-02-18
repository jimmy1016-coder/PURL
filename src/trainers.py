import math
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class NCUTrainer(SupervisedTrainer):
    """Noisy Correspondence Unlearning Trainer.

    매 에폭 시작 시 현재 모델로 전체 embedding을 추출하고,
    화자별 GMM으로 clean/noisy pair를 판별한 뒤,
    ncu_loss_type에 따라 다른 전략으로 학습한다.

    ncu_loss_type:
        "hard": clean/noisy를 threshold(0.5) 기준으로 이분화.
                Clean → SupCon, Noisy → cosine repulsion.
        "soft": GMM posterior p_clean을 soft weight로 사용.
                Confident clean → SupCon, 모든 pair → (1-p_clean) weighted repulsion.
    """

    def __init__(
        self,
        encoder,
        projector,
        loss_func: nn.Module,
        learning_rate: float,
        lr_scheduler_type: str,
        sample_rate: int,
        alpha: float = 0.1,
        ncu_loss_type: str = "hard",
        ncu_clean_threshold: float = 0.5,
        ncu_labels_path: str = None,
        ncu_data_dir: str = "",
        ncu_spk2utt_file_path: str = "",
        ncu_gmm_every_n_epochs: int = 1,
        ncu_batch_size: int = 64,
        ncu_num_workers: int = 4,
        optim_weight_decay: float = None,
        samples_per_epoch: int = None,
        batch_size: int = None,
        clustering_model=None,
    ):
        super().__init__(
            encoder=encoder,
            projector=projector,
            loss_func=loss_func,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            sample_rate=sample_rate,
            optim_weight_decay=optim_weight_decay,
            samples_per_epoch=samples_per_epoch,
            batch_size=batch_size,
            clustering_model=clustering_model,
        )
        assert ncu_loss_type in ("hard", "soft"), f"Unknown ncu_loss_type: {ncu_loss_type}"
        self.alpha = alpha
        self.ncu_loss_type = ncu_loss_type
        self.ncu_clean_threshold = ncu_clean_threshold
        self.ncu_labels_path = ncu_labels_path
        self.ncu_data_dir = ncu_data_dir
        self.ncu_spk2utt_file_path = ncu_spk2utt_file_path
        self.ncu_gmm_every_n_epochs = ncu_gmm_every_n_epochs
        self.ncu_batch_size = ncu_batch_size
        self.ncu_num_workers = ncu_num_workers

    def on_train_epoch_start(self):
        """에폭 시작 시 현재 모델로 clean/noisy labels를 재생성한다.

        - 시작 에폭(0 또는 200 등): 무조건 GMM fitting
        - 이후: ncu_gmm_every_n_epochs 주기마다 수행 (예: 30이면 200 → 230 → 260 ...)
        """
        if not hasattr(self, "_ncu_gmm_start_epoch"):
            self._ncu_gmm_start_epoch = self.current_epoch

        start = self._ncu_gmm_start_epoch
        ep = self.current_epoch
        n = self.ncu_gmm_every_n_epochs

        is_start_epoch = ep == start
        is_periodic = ep > start and (ep - start) % n == 0

        if not is_start_epoch and not is_periodic:
            next_update = start + ((ep - start) // n + 1) * n
            print(f"[NCU] Epoch {ep}: Skipping GMM fitting (next at epoch {next_update})", flush=True)
            return

        import ncu_utils
        from data.data_utils import load_spk2utt

        reason = "start epoch" if is_start_epoch else f"periodic (every {n} epochs)"
        print(f"\n[NCU] Epoch {ep}: Updating clean/noisy labels ({reason})...", flush=True)

        spk2utt = load_spk2utt(self.ncu_spk2utt_file_path)

        # 1. 현재 모델로 전체 embedding 추출
        self.encoder.eval()
        embeddings_dict = ncu_utils.extract_all_embeddings(
            encoder=self.encoder,
            spk2utt=spk2utt,
            data_dir=self.ncu_data_dir,
            device=self.device,
            batch_size=self.ncu_batch_size,
            num_workers=self.ncu_num_workers,
        )
        self.encoder.train()

        # 2. 화자별 GMM fit
        gmm_dict = ncu_utils.fit_per_speaker_gmm(spk2utt, embeddings_dict)

        # 3. Pickle로 저장 (덮어쓰기)
        ncu_utils.save_ncu_labels(self.ncu_labels_path, embeddings_dict, gmm_dict)

        # 통계 로그
        n_speakers_with_gmm = sum(1 for v in gmm_dict.values() if v is not None)
        print(f"[NCU] GMM fitted for {n_speakers_with_gmm}/{len(gmm_dict)} speakers")
        print(f"[NCU] Labels saved to {self.ncu_labels_path}")

        # Dataset에 labels reload 트리거
        try:
            train_dl = self.trainer.train_dataloader
            if train_dl is not None:
                dataset = train_dl.dataset
                if hasattr(dataset, '_load_ncu_labels'):
                    dataset._load_ncu_labels()
                    print("[NCU] Dataset labels reloaded in main process")
        except Exception as e:
            print(f"[NCU] Warning: Could not reload dataset labels: {e}")

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
        features, audios, labels, p_clean = batch

        if self.global_step < 5 and self.global_rank == 0:
            self._log_audio(audios, "train")

        features1 = features[:, 0]
        features2 = features[:, 1]

        # Encoder forward (전체 배치, 한 번만)
        embs = self.encoder(
            torch.cat([features1, features2], dim=0)
        )
        embs = self.projector(embs)

        emb1, emb2 = torch.split(embs, features.shape[0])

        if self.ncu_loss_type == "hard":
            loss, loss_clean_val, loss_noisy_val, n_clean, n_noisy = self._training_step_hard(
                emb1, emb2, labels, p_clean, features.device
            )
        else:  # soft
            loss, loss_clean_val, loss_noisy_val, n_clean, n_noisy = self._training_step_soft(
                emb1, emb2, labels, p_clean, features.device
            )

        # Logging
        batch_size = features.shape[0]
        self.log("train/loss_total", float(loss.detach()), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/loss_clean", loss_clean_val, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/loss_noisy", loss_noisy_val, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/n_clean", float(n_clean), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/n_noisy", float(n_noisy), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/clean_ratio", float(n_clean) / max(n_clean + n_noisy, 1), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/alpha", self.alpha, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        return loss

    def _training_step_hard(self, emb1, emb2, labels, p_clean, device):
        """Loss 2: Hard split — clean/noisy를 threshold 기준으로 이분화."""
        clean_mask = (p_clean > self.ncu_clean_threshold)
        noisy_mask = ~clean_mask

        n_clean = clean_mask.sum().item()
        n_noisy = noisy_mask.sum().item()

        loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_clean_val = 0.0
        loss_noisy_val = 0.0

        # Clean set → SupCon loss
        if n_clean > 1:
            loss_clean, metrics_clean = self.loss_func(
                emb1[clean_mask].squeeze(1),
                emb2[clean_mask].squeeze(1),
                labels[clean_mask],
            )
            loss = loss + loss_clean
            loss_clean_val = metrics_clean.get("loss", 0.0)

        # Noisy set → pairwise cosine similarity 최소화
        if n_noisy > 0:
            emb1_noisy = F.normalize(emb1[noisy_mask].squeeze(1), dim=1)
            emb2_noisy = F.normalize(emb2[noisy_mask].squeeze(1), dim=1)
            noisy_sim = (emb1_noisy * emb2_noisy).sum(dim=1)
            loss_noisy_repulsion = noisy_sim.mean()
            loss = loss + self.alpha * loss_noisy_repulsion
            loss_noisy_val = loss_noisy_repulsion.item()

        return loss, loss_clean_val, loss_noisy_val, n_clean, n_noisy

    def _training_step_soft(self, emb1, emb2, labels, p_clean, device):
        """Loss 3: Soft weighting — p_clean을 연속 weight로 사용."""
        clean_mask = (p_clean > self.ncu_clean_threshold)
        w_noisy = (1.0 - p_clean)  # [batch_size], 0이면 확실한 clean, 1이면 확실한 noisy

        n_clean = clean_mask.sum().item()
        n_noisy = (~clean_mask).sum().item()

        loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_clean_val = 0.0
        loss_noisy_val = 0.0

        # Confident clean → SupCon loss (p_clean > threshold인 pair만)
        if n_clean > 1:
            loss_clean, metrics_clean = self.loss_func(
                emb1[clean_mask].squeeze(1),
                emb2[clean_mask].squeeze(1),
                labels[clean_mask],
            )
            loss = loss + loss_clean
            loss_clean_val = metrics_clean.get("loss", 0.0)

        # 모든 pair → (1 - p_clean) weighted cosine repulsion
        emb1_norm = F.normalize(emb1.squeeze(1), dim=1)
        emb2_norm = F.normalize(emb2.squeeze(1), dim=1)
        cos_sim = (emb1_norm * emb2_norm).sum(dim=1)  # [batch_size]
        loss_noisy_repulsion = (w_noisy * cos_sim).mean()
        loss = loss + self.alpha * loss_noisy_repulsion
        loss_noisy_val = loss_noisy_repulsion.item()

        return loss, loss_clean_val, loss_noisy_val, n_clean, n_noisy


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
