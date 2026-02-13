import torch
import numpy as np
import pickle
import os
import pandas as pd

def load_data(graph_dir, data_path):
    # 加载图结构
    G_pet_path = os.path.join(graph_dir, "G_pet.pkl")
    G_pathology_path = os.path.join(graph_dir, "G_pathology.pkl")
    G_gene_path = os.path.join(graph_dir, "G_gene.pkl")
    G_extra_path = os.path.join(graph_dir, "G_extra.pkl")

    with open(G_pet_path, "rb") as f:
        G_pet = pickle.load(f)
    with open(G_pathology_path, "rb") as f:
        G_pathology = pickle.load(f)
    with open(G_gene_path, "rb") as f:
        G_gene = pickle.load(f)
    with open(G_extra_path, "rb") as f:
        G_extra = pickle.load(f)  # 加载 G_extra 图

    pet_edge_index = torch.tensor(list(G_pet.edges()), dtype=torch.long).T
    pathology_edge_index = torch.tensor(list(G_pathology.edges()), dtype=torch.long).T
    gene_edge_index = torch.tensor(list(G_gene.edges()), dtype=torch.long).T
    extra_edge_index = torch.tensor(list(G_extra.edges()), dtype=torch.long).T  # 提取 G_extra 的边索引

    def get_embedding_dim(G, embedding_key):
        for node in G.nodes():
            if embedding_key in G.nodes[node]:
                return G.nodes[node][embedding_key].shape[0]
        raise ValueError(f"No node has '{embedding_key}' attribute. Cannot determine embedding dimension.")

    pet_embedding_dim = get_embedding_dim(G_pet, "features")
    pathology_embedding_dim = get_embedding_dim(G_pathology, "features")
    gene_embedding_dim = get_embedding_dim(G_gene, "features")
    extra_embedding_dim = get_embedding_dim(G_extra, "features")  # 确定 G_extra 的嵌入维度

    X_pet = torch.tensor(
        np.array([
            G_pet.nodes[node].get("features", np.full(pet_embedding_dim, np.nan))
            for node in G_pet.nodes()
        ]),
        dtype=torch.float
    )
    X_pathology = torch.tensor(
        np.array([
            G_pathology.nodes[node].get("features", np.full(pathology_embedding_dim, np.nan))
            for node in G_pathology.nodes()
        ]),
        dtype=torch.float
    )
    X_gene = torch.tensor(
        np.array([
            G_gene.nodes[node].get("features", np.full(gene_embedding_dim, np.nan))
            for node in G_gene.nodes()
        ]),
        dtype=torch.float
    )
    X_extra = torch.tensor(
        np.array([
            G_extra.nodes[node].get("features", np.full(extra_embedding_dim, np.nan))
            for node in G_extra.nodes()
        ]),
        dtype=torch.float
    )

    df = pd.read_excel(data_path)
    y = torch.tensor(df['是否复发（1是 0否）'].values, dtype=torch.long)

    return X_pet, X_pathology, X_gene, X_extra, pet_edge_index, pathology_edge_index, gene_edge_index, extra_edge_index, y