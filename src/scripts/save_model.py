import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from trainers import srcTrainer
from models import ECAPAEmbeddingModel
from losses import NTXentLoss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
    )
    parser.add_argument("--ckpt-name", type=str, required=True)

    args = parser.parse_args()

    output_dir_path = Path(args.experiment_dir) / "embedding_model_checkpoints"
    output_dir_path.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.load(Path(args.experiment_dir) / "config.yaml")

    embedding_model = ECAPAEmbeddingModel(
        **config.model.init_args.embedding_model.init_args
    )
    loss_func = NTXentLoss(**config.model.init_args.loss_func.init_args)

    model = srcTrainer.load_from_checkpoint(
        checkpoint_path=Path(args.experiment_dir)
        / "checkpoints"
        / (args.ckpt_name + ".ckpt"),
        embedding_model=embedding_model,
        loss_func=loss_func,
    )

    output_ckpt_path = output_dir_path / (args.ckpt_name + ".pt")
    torch.save(model.embedding_model.model.state_dict(), output_ckpt_path)
