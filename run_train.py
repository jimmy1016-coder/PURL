from pathlib import Path
import logging

from setproctitle import setproctitle
from lightning.pytorch.cli import LightningCLI


logger = logging.getLogger("lightning")

if __name__ == "__main__":
    cli = LightningCLI(run=False, save_config_kwargs={"overwrite": True}, parser_kwargs={"parser_mode": "omegaconf"})

    """
    If the output dir already exists and has a "last.ckpt" checkpoint,
    load this checkpoint and continue training.
    """
    output_dir = Path(cli.trainer.logger.save_dir)
    setproctitle(output_dir.name)

    last_ckpt_path = output_dir / "checkpoints" / "last.ckpt"
    ckpt_load_path = None

    if last_ckpt_path.is_file():
        logger.info(f"Loading existing training state from: {last_ckpt_path}")
        ckpt_load_path = last_ckpt_path

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_load_path)
