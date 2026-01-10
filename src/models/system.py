import lightning as L
import torch
import torch.nn.functional as F
from src.models.changenet_mlp import ChangeNet

class ChangeDetectionSystem(L.LightningModule):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = ChangeNet(in_dim=in_dim, hidden_dim=hidden_dim)

    def forward(self, P, Q, mask_p, mask_q):
        return self.model(P, Q, mask_p=mask_p, mask_q=mask_q)

    def _shared_step(self, batch):
        pred_q_logits, pred_p_logits = self(batch["P"], batch["Q"], batch["mask_p"], batch["mask_q"])
        
        loss_q = F.binary_cross_entropy_with_logits(pred_q_logits, batch["change_q"], reduction='none')
        loss_p = F.binary_cross_entropy_with_logits(pred_p_logits, batch["change_p"], reduction='none')

        masked_loss = ((loss_q * batch["mask_q"]).sum() / batch["mask_q"].sum()) + \
                      ((loss_p * batch["mask_p"]).sum() / batch["mask_p"].sum())
        return masked_loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)