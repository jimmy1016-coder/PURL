from pathlib import Path
import logging

from setproctitle import setproctitle
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger


logger = logging.getLogger("lightning")


class TestLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_name", required=True)
        parser.add_argument("--no_wandb", action="store_true", help="Disable wandb during test")


if __name__ == "__main__":
    cli = TestLightningCLI(
        run=False, save_config_kwargs={"overwrite": True}, parser_kwargs={"parser_mode": "omegaconf"}
    )

    if getattr(cli.config, "no_wandb", False):
        loggers = cli.trainer.loggers
        save_dir = cli.trainer.logger.save_dir
        if isinstance(loggers, list):
            new_loggers = [lg for lg in loggers if not isinstance(lg, WandbLogger)]
            if not new_loggers:
                new_loggers = [TensorBoardLogger(save_dir=save_dir)]
        elif isinstance(loggers, WandbLogger):
            new_loggers = [TensorBoardLogger(save_dir=save_dir)]
        else:
            new_loggers = [loggers]
        cli.trainer._loggers = new_loggers

    output_dir = Path(cli.trainer.logger.save_dir)
    setproctitle(f"{output_dir.name}-test")

    ckpt_path = output_dir / "checkpoints" / f"{cli.config.ckpt_name}.ckpt"

    cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
