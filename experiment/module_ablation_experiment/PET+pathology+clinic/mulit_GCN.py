import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN_model.models import get_model  # 确保这个路径正确
from AutoTriModalFuse import AutoDimFuse


# 主模型
class Multi_GNN_model(nn.Module):
    def __init__(self, pet_num_features, pathology_num_features, extra_num_features,
                 num_classes, pet_edge_index, pathology_edge_index, extra_edge_index,
                 pet_x, pathology_x, extra_x, args):
        super(Multi_GNN_model, self).__init__()

        # 初始化三个 GCN 模型
        self.gcn_pet = get_model("gcnmf", pet_num_features, num_classes, pet_edge_index, pet_x, args)
        self.gcn_pathology = get_model("gcnmf", pathology_num_features, num_classes, pathology_edge_index, pathology_x, args)

        self.gcn_extra = get_model("gcnmf", extra_num_features, num_classes, extra_edge_index, extra_x, args)

        # 1. 单瓶颈层 + 正则
        self.bottleneck = nn.Sequential(
            nn.Linear(args.hidden_dim*4, args.hidden_dim*4),
            nn.LayerNorm(args.hidden_dim*4),
            nn.Dropout(args.post_fusion_dropout)
        )

        self.fuse = AutoDimFuse(dim=args.hidden_dim)

        # 分类器
        self.classifier = nn.Linear(args.hidden_dim*4, num_classes)

    def forward(self, pet_x, pathology_x, extra_x, pet_edge_index, pathology_edge_index, extra_edge_index):
        # GCN 前向传播，输出为 [num_nodes, hidden_dim]
        pet_log, pet_x_gcn = self.gcn_pet(pet_x, pet_edge_index)
        pathology_log, pathology_x_gcn = self.gcn_pathology(pathology_x, pathology_edge_index)

        extra_log, extra_x_gcn = self.gcn_extra(extra_x, extra_edge_index)

        x_f = self.fuse([pet_x_gcn, pathology_x_gcn, extra_x_gcn])
        # x_f = self.bottleneck(fusion_h_3)
        x_f_4 = torch.cat([pet_x_gcn, pathology_x_gcn,extra_x_gcn, x_f], dim=-1)
        x_f = self.bottleneck(x_f_4)
        # 对每个节点进行分类
        logits = self.classifier(x_f)

        # 融合边索引 + 过滤
        fused_edge_index = self.merge_edge_indices(pet_edge_index, pathology_edge_index, extra_edge_index)

        return logits, x_f, fused_edge_index

    def merge_edge_indices(self, edge_index1, edge_index2, edge_index3):
        """
        合并四个模态的边索引，并保留重复边中的一条。

        参数:
            edge_index1, edge_index2, edge_index3, edge_index4: Tensor, shape [2, num_edges]

        返回:
            filtered_edge_index: Tensor, shape [2, num_filtered_edges]
        """

        # 合并四个模态的边
        fused_edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)

        # 将边索引转换为二维张量，每行表示一条边
        edges = torch.cat([fused_edge_index[0].unsqueeze(1), fused_edge_index[1].unsqueeze(1)], dim=1)

        # 去除重复的边，只保留一条
        unique_edges, unique_indices = torch.unique(edges, dim=0, return_inverse=True)

        # 将去重后的边重新转换为边索引格式
        filtered_edge_index = unique_edges.t()  # 转置回 [2, num_unique_edges]

        return filtered_edge_index

    # def merge_edge_indices(self, edge_index1, edge_index2, edge_index3):
    #     """
    #     合并三个模态的边索引，并保留重复边中的一条。
    #
    #     参数:
    #         edge_index1, edge_index2, edge_index3: Tensor, shape [2, num_edges]
    #
    #     返回:
    #         filtered_edge_index: Tensor, shape [2, num_filtered_edges]
    #     """
    #
    #     # 合并三个模态的边
    #     fused_edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)
    #
    #     # 将边索引转换为二维张量，每行表示一条边
    #     edges = torch.cat([fused_edge_index[0].unsqueeze(1), fused_edge_index[1].unsqueeze(1)], dim=1)
    #
    #     # 去除重复的边，只保留一条
    #     unique_edges, unique_indices = torch.unique(edges, dim=0, return_inverse=True)
    #
    #     # 将去重后的边重新转换为边索引格式
    #     filtered_edge_index = unique_edges.t()  # 转置回 [2, num_unique_edges]
    #
    #     return filtered_edge_index