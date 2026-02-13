import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoDimFuse(nn.Module):
    """
    Input: 4 tensors of same shape [B, D]
    Output: Fused [B, D], each dimension selects only one modality (dim-wise selection)
    """
    def __init__(self, dim):
        super().__init__()

        # MLP generates [B, D, 4] logits
        self.gate = nn.Sequential(
            nn.Linear(dim * 4, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim * 4)
        )

    def forward(self, x_list, return_ratio=False):
        """
        x_list: [x0, x1, x2, x3], each [B, D]
        """
        assert len(x_list) == 4
        x0, x1, x2, x3 = x_list   # [B, D]

        # Concatenate input → [B, 4D]
        z = torch.cat([x0, x1, x2, x3], dim=-1)

        # gate output [B, 4D] → reshape to [B, D, 4]
        logits = self.gate(z).view(z.size(0), z.size(1) // 4, 4)

        # =======================
        #   Training phase: Gumbel-Softmax (backpropagation enabled)
        # =======================
        if self.training:
            mask = F.gumbel_softmax(logits, tau=1.0, hard=True)  # [B, D, 4] one-hot
        else:
            # =======================
            #   Inference phase: argmax
            # =======================
            idx = torch.argmax(logits, dim=-1)   # [B, D]
            mask = F.one_hot(idx, num_classes=4).float()  # [B, D, 4]

        # Stack four modalities to [B, D, 4]
        stacked = torch.stack(x_list, dim=-1)

        # Select modality for each dimension based on one-hot mask
        fused = torch.sum(stacked * mask, dim=-1)  # [B, D]

        # Calculate modality ratio (inference phase only)
        if return_ratio:
            modality_ratio = self.get_modality_usage_ratio(mask)
            return fused, modality_ratio

        return fused

    def get_modality_usage_ratio(self, mask):
        """
        mask: [B, D, 4] one-hot
        Returns: list of length 4, representing the proportion of each modality being selected
        """
        mask = mask.reshape(-1, 4)  # [B*D, 4]
        total = mask.size(0)
        ratios = (mask.sum(dim=0) / total).tolist()
        return ratios