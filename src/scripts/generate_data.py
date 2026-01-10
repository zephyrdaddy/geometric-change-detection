import hydra
from omegaconf import DictConfig
from src.utils.file_io import generate_hdf5_dataset, count_hdf5_samples

@hydra.main(version_base="1.3", config_path="../../configs/data_generation", config_name="default")
def main(cfg: DictConfig):
    # Use Hydra's utility to make sure the output path is absolute
    output_path = hydra.utils.to_absolute_path(cfg.output_dir)
    
    generate_hdf5_dataset(
        n_samples=cfg.n_samples,
        output_dir=output_path,
        batch_size=cfg.batch_size,
        n_points=cfg.n_points,
        overwrite=cfg.overwrite,
        noise_std=cfg.noise_std,
    )

    total = count_hdf5_samples(output_path)
    print(f"\n✅ Finished. Total samples in {output_path}: {total}")

if __name__ == "__main__":
    main()