import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GNN_model.models import get_model  # 确保这个路径正确



class Multi_GNN_model(nn.Module):
    def __init__(self, pet_num_features, pathology_num_features, gene_num_features, extra_num_features,
                 num_classes, pet_edge_index, pathology_edge_index, gene_edge_index, extra_edge_index,
                 pet_x, pathology_x, gene_x, extra_x, args):
        super(Multi_GNN_model, self).__init__()

        self.gcn_pet = get_model("gcnmf", pet_num_features, num_classes, pet_edge_index, pet_x, args)
        self.gcn_pathology = get_model("gcnmf", pathology_num_features, num_classes, pathology_edge_index, pathology_x, args)
        self.gcn_gene = get_model("gcnmf", gene_num_features, num_classes, gene_edge_index, gene_x, args)
        self.gcn_extra = get_model("gcnmf", extra_num_features, num_classes, extra_edge_index, extra_x, args)

        self.bottleneck = nn.Sequential(
            nn.Linear(args.hidden_dim*4, args.hidden_dim*4),
            nn.LayerNorm(args.hidden_dim*4),
            nn.Dropout(args.post_fusion_dropout)
        )

        self.classifier = nn.Linear(args.hidden_dim*4, num_classes)

    def forward(self, pet_x, pathology_x, gene_x, extra_x, pet_edge_index, pathology_edge_index, gene_edge_index, extra_edge_index):
        pet_log, pet_x_gcn = self.gcn_pet(pet_x, pet_edge_index)
        pathology_log, pathology_x_gcn = self.gcn_pathology(pathology_x, pathology_edge_index)
        gene_log, gene_x_gcn = self.gcn_gene(gene_x, gene_edge_index)
        extra_log, extra_x_gcn = self.gcn_extra(extra_x, extra_edge_index)

        x_f_4 = torch.cat([pet_x_gcn, pathology_x_gcn, gene_x_gcn, extra_x_gcn], dim=-1)
        x_f = self.bottleneck(x_f_4)
        logits = self.classifier(x_f)

        fused_edge_index = self.merge_edge_indices(pet_edge_index, pathology_edge_index, gene_edge_index, extra_edge_index)

        return logits, pet_log, pathology_log, gene_log, extra_log, x_f, fused_edge_index

    def merge_edge_indices(self, edge_index1, edge_index2, edge_index3, edge_index4):


        fused_edge_index = torch.cat([edge_index1, edge_index2, edge_index3, edge_index4], dim=1)


        edges = torch.cat([fused_edge_index[0].unsqueeze(1), fused_edge_index[1].unsqueeze(1)], dim=1)


        unique_edges, unique_indices = torch.unique(edges, dim=0, return_inverse=True)


        filtered_edge_index = unique_edges.t()

        return filtered_edge_index

