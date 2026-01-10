import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint

# HYDRA_CONFIG_DIR=""
@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Instantiate using Hydra
    datamodule = hydra.utils.instantiate(cfg.data)
    system = hydra.utils.instantiate(cfg.model)
    
    # Best model saving logic (standard PL)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best-change-model-{epoch:02d}",
        save_top_k=1,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback]
    )

    trainer.fit(system, datamodule=datamodule)

if __name__ == "__main__":
    main()