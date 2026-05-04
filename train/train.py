import logging

import hydra
import torch
import pytorch_lightning

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')
pytorch_lightning.seed_everything(333)


@hydra.main(config_path="config", config_name="main", version_base="1.3")
def train(cfg):
    trainer = hydra.utils.instantiate(cfg.trainer)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)

    if trainer.logger is not None:
        logger.info(f"Log directory: {trainer.logger.log_dir}")

    if cfg.finetune:
        state_dict = torch.load(cfg.finetune, weights_only=True)['state_dict']
        model.load_state_dict(state_dict)

        if trainer.logger is not None:
            logger.info(f"Finetuning from checkpoint {cfg.finetune}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg['ckpt'])


if __name__ == "__main__":
    train()
