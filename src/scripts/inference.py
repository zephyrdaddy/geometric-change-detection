import hydra
import torch
import h5py
import numpy as np
from omegaconf import DictConfig
from src.models.system import ChangeDetectionSystem

@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    # 1. Load Model from Lightning Checkpoint
    # This automatically restores hyperparameters and architecture
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # Use the class method to load the weights
    model = ChangeDetectionSystem.load_from_checkpoint(
        checkpoint_path=hydra.utils.to_absolute_path(cfg.model_ckpt)
    ).to(device)
    model.eval()

    input_path = hydra.utils.to_absolute_path(cfg.input_h5)
    output_path = hydra.utils.to_absolute_path(cfg.output_h5)

    with h5py.File(input_path, 'r') as fin, h5py.File(output_path, 'w') as fout:
        for key in fin.keys():
            grp = fin[key]
            # Convert to Tensors and add Batch Dim
            P = torch.from_numpy(np.array(grp['P'])).float().unsqueeze(0).to(device)
            Q = torch.from_numpy(np.array(grp['Q'])).float().unsqueeze(0).to(device)
            
            # Forward pass (LightningModule forward usually calls the internal model)
            with torch.no_grad():
                # We assume ChangeDetectionSystem.forward returns (logits_q, logits_p)
                # Ensure your model handles the dummy mask input if your forward requires it
                logits_q, logits_p = model(P, Q, mask_p=None, mask_q=None)
                
                pred_p = torch.sigmoid(logits_p).cpu().numpy().squeeze()
                pred_q = torch.sigmoid(logits_q).cpu().numpy().squeeze()

            # Save results
            out_grp = fout.create_group(key)
            for dset in ['P', 'Q', 'change_p', 'change_q']:
                out_grp.create_dataset(dset, data=grp[dset])
            
            out_grp.create_dataset('change_p_pred', data=pred_p)
            out_grp.create_dataset('change_q_pred', data=pred_q)

    print(f"✅ Inference complete. Results saved to: {output_path}")

if __name__ == "__main__":
    main()