from pathlib import Path
import logging
import sys

from setproctitle import setproctitle
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger


logger = logging.getLogger("lightning")

if __name__ == "__main__":
    # --ckpt_path는 LightningCLI가 인식하지 못하므로, 먼저 추출 후 제거
    ckpt_path_arg = None
    if '--ckpt_path' in sys.argv:
        idx = sys.argv.index('--ckpt_path')
        ckpt_path_arg = sys.argv[idx + 1]
        del sys.argv[idx:idx + 2]

    cli = LightningCLI(run=False, save_config_kwargs={"overwrite": True}, parser_kwargs={"parser_mode": "omegaconf"})

    """
    If the output dir already exists and has a "last.ckpt" checkpoint,
    load this checkpoint and continue training.
    """
    output_dir = Path(cli.trainer.logger.save_dir)
    setproctitle(output_dir.name)

    # Configure wandb logger if present
    wandb_logger = None
    if isinstance(cli.trainer.logger, WandbLogger):
        wandb_logger = cli.trainer.logger
    elif isinstance(cli.trainer.logger, list):
        for logger_item in cli.trainer.logger:
            if isinstance(logger_item, WandbLogger):
                wandb_logger = logger_item
                break
    
    if wandb_logger is not None:
        # Log model hyperparameters to wandb
        if hasattr(cli.model, 'hparams'):
            try:
                # Convert hyperparameters to dict if needed
                hparams = dict(cli.model.hparams) if hasattr(cli.model.hparams, 'keys') else cli.model.hparams
                wandb_logger.log_hyperparams(hparams)
            except Exception as e:
                logger.warning(f"Could not log model hyperparameters to wandb: {e}")
        
        # Log important config values
        try:
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(cli.config, resolve=True)
            
            important_config = {}
            if 'model' in config_dict and 'init_args' in config_dict['model']:
                model_args = config_dict['model']['init_args']
                important_config['learning_rate'] = model_args.get('learning_rate', 'N/A')
                important_config['optim_weight_decay'] = model_args.get('optim_weight_decay', 'N/A')
                important_config['lr_scheduler_type'] = model_args.get('lr_scheduler_type', 'N/A')
                
                if 'loss_func' in model_args and 'init_args' in model_args['loss_func']:
                    loss_args = model_args['loss_func']['init_args']
                    important_config['temperature'] = loss_args.get('temperature', 'N/A')
                    important_config['learn_temperature'] = loss_args.get('learn_temperature', 'N/A')
                    important_config['margin'] = loss_args.get('margin', 'N/A')
            
            if 'trainer' in config_dict:
                important_config['max_epochs'] = config_dict['trainer'].get('max_epochs', 'N/A')
                important_config['batch_size'] = config_dict.get('data', {}).get('init_args', {}).get('train_config', {}).get('batch_size', 'N/A')
            
            if important_config:
                wandb_logger.log_hyperparams(important_config)
        except Exception as e:
            logger.warning(f"Could not log config to wandb: {e}")
        
        logger.info("Wandb logger configured. Metrics will be logged to wandb.")

    # Determine checkpoint path: CLI arg > auto-resume from last.ckpt
    ckpt_load_path = ckpt_path_arg

    if ckpt_load_path is not None:
        logger.info(f"Resuming training from checkpoint specified via --ckpt_path: {ckpt_load_path}")
    else:
        # Auto-resume: look for last.ckpt in checkpoints directory
        last_ckpt_path = output_dir / "checkpoints" / "last.ckpt"
        if last_ckpt_path.is_file():
            logger.info(f"Auto-resuming: Loading existing training state from: {last_ckpt_path}")
            ckpt_load_path = str(last_ckpt_path)

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_load_path)
