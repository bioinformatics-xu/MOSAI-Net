import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN_model.models import get_model  # 确保这个路径正确
from model.AMSF.AutoTriModalFuse import AutoDimFuse


class Multi_GNN_model(nn.Module):
    def __init__(self, extra_num_features,
                 num_classes, extra_edge_index,
                 extra_x, args):
        super(Multi_GNN_model, self).__init__()

        self.gcn_extra = get_model("gcnmf", extra_num_features, num_classes, extra_edge_index, extra_x, args)

        self.bottleneck = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.Dropout(args.post_fusion_dropout)
        )

        self.classifier = nn.Linear(args.hidden_dim, num_classes)

    def forward(self, extra_x, extra_edge_index):
        extra_log, extra_x_gcn = self.gcn_extra(extra_x, extra_edge_index)

        x_f = self.bottleneck(extra_x_gcn)
        logits = self.classifier(x_f)


        return logits, extra_log, x_f, extra_edge_index

    def merge_edge_indices(self, edge_index1, edge_index2, edge_index3, edge_index4):

        fused_edge_index = torch.cat([edge_index1, edge_index2, edge_index3, edge_index4], dim=1)

        edges = torch.cat([fused_edge_index[0].unsqueeze(1), fused_edge_index[1].unsqueeze(1)], dim=1)

        unique_edges, unique_indices = torch.unique(edges, dim=0, return_inverse=True)

        filtered_edge_index = unique_edges.t()

        return filtered_edge_index

