from pathlib import Path
import logging

from setproctitle import setproctitle
from lightning.pytorch.cli import LightningCLI


logger = logging.getLogger("lightning")


class TestLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--ckpt_name", required=True)


if __name__ == "__main__":
    cli = TestLightningCLI(
        run=False, save_config_kwargs={"overwrite": True}, parser_kwargs={"parser_mode": "omegaconf"}
    )

    output_dir = Path(cli.trainer.logger.save_dir)
    setproctitle(f"{output_dir.name}-test")

    ckpt_path = output_dir / "checkpoints" / f"{cli.config.ckpt_name}.ckpt"

    cli.trainer.test(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_path)
