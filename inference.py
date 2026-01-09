import torch
import h5py
import numpy as np
from pathlib import Path
from models.changenet_mlp import ChangeNet # Adjust path as needed

def run_inference(model_path, input_h5, output_h5, device='cuda'):
    # 1. Load Model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = ChangeNet(in_dim=2, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with h5py.File(input_h5, 'r') as fin, h5py.File(output_h5, 'w') as fout:
        for key in fin.keys():
            # Load data
            grp = fin[key]
            P = torch.from_numpy(np.array(grp['P'])).float().unsqueeze(0).to(device) # [1, n_p, 2]
            Q = torch.from_numpy(np.array(grp['Q'])).float().unsqueeze(0).to(device) # [1, n_q, 2]
            
            # Forward pass
            with torch.no_grad():
                logits_q, logits_p = model(P, Q)
                pred_p = torch.sigmoid(logits_p).cpu().numpy().squeeze()
                pred_q = torch.sigmoid(logits_q).cpu().numpy().squeeze()

            # Save to new file
            out_grp = fout.create_group(key)
            out_grp.create_dataset('P', data=grp['P'])
            out_grp.create_dataset('Q', data=grp['Q'])
            out_grp.create_dataset('change_p', data=grp['change_p'])
            out_grp.create_dataset('change_q', data=grp['change_q'])
            out_grp.create_dataset('change_p_pred', data=pred_p)
            out_grp.create_dataset('change_q_pred', data=pred_q)

    print(f"✅ Inference complete. Results saved to {output_h5}")

if __name__ == "__main__":
    run_inference("best_change_model.pth", "data/generated/change_dataset_batch_0009.h5", "results.h5")