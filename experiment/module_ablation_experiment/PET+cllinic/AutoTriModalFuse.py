import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoDimFuse(nn.Module):
    """
    输入: 2 个同形张量 [B, D]（D 为特征维度）
    输出: [B, D]  每个维度独立地选 0/1 中分数最高的
    """
    def __init__(self, dim):
        super().__init__()
        # 为每个维度独立预测 2 类分数
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),  # 先升维
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2)   # 输出 [B, D*2]
        )

    def forward(self, x_list, return_ratio=False):
        """
        x_list: [x0, x1], 每个 [B, D]
        return_ratio: 是否返回模态使用比例（仅推理时有效）
        """
        assert len(x_list) == 2
        x0, x1 = x_list                      # [B, D]
        z = torch.cat([x0, x1], dim=-1)      # [B, 2D]
        logits = self.gate(z).view(x0.size(0), x0.size(1), 2)  # [B, D, 2]

        # =======================
        #   训练阶段：Gumbel-Softmax（可反向传播）
        # =======================
        if self.training:
            mask = F.gumbel_softmax(logits, tau=5.0, hard=True)  # [B, D, 2] one-hot
        else:
            # =======================
            #   推理阶段：argmax
            # =======================
            idx = torch.argmax(logits, dim=-1)                   # [B, D]
            mask = F.one_hot(idx, num_classes=2).float()         # [B, D, 2]

        # 将两模态堆叠为 [B, D, 2]
        stacked = torch.stack(x_list, dim=-1)

        # 基于 one-hot mask 选择每个维度的模态（可微操作）
        fused = torch.sum(stacked * mask, dim=-1)  # [B, D]

        # 统计模态占比（仅推理阶段）
        if return_ratio and not self.training:
            modality_ratio = self.get_modality_usage_ratio(mask)
            return fused, modality_ratio

        return fused

    def get_modality_usage_ratio(self, mask):
        """
        mask: [B, D, 2] one-hot
        返回：list 长度 2，分别是模态 0 和模态 1 被选中的比例
        """
        mask = mask.reshape(-1, 2)  # [B*D, 2]
        total = mask.size(0)
        ratios = (mask.sum(dim=0) / total).tolist()
        return ratios