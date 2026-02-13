import torch
import numpy as np
import pickle
import os
import pandas as pd

def load_data(graph_dir, data_path):
    # 加载图结构
    G_gene_path = os.path.join(graph_dir, "G_gene.pkl")

    with open(G_gene_path, "rb") as f:
        G_gene = pickle.load(f)

    # 提取边索引
    gene_edge_index = torch.tensor(list(G_gene.edges()), dtype=torch.long).T

    # 确定嵌入维度
    def get_embedding_dim(G, embedding_key):
        for node in G.nodes():
            if embedding_key in G.nodes[node]:
                return G.nodes[node][embedding_key].shape[0]
        raise ValueError(f"No node has '{embedding_key}' attribute. Cannot determine embedding dimension.")

    gene_embedding_dim = get_embedding_dim(G_gene, "features")

    # 提取节点嵌入特征，保留 NaN 值，没有 "features" 属性的节点用全为 NaN 的向量填充
    X_gene = torch.tensor(
        np.array([
            G_gene.nodes[node].get("features", np.full(gene_embedding_dim, np.nan))
            for node in G_gene.nodes()
        ]),
        dtype=torch.float
    )

    # 加载标签数据
    df = pd.read_excel(data_path)
    y = torch.tensor(df['是否复发（1是 0否）'].values, dtype=torch.long)

    return X_gene,  gene_edge_index, y