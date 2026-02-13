import torch
import numpy as np
import pickle
import os
import pandas as pd

def load_data(graph_dir, data_path):
    G_extra_path = os.path.join(graph_dir, "G_extra.pkl")

    with open(G_extra_path, "rb") as f:
        G_extra = pickle.load(f)

    extra_edge_index = torch.tensor(list(G_extra.edges()), dtype=torch.long).T

    def get_embedding_dim(G, embedding_key):
        for node in G.nodes():
            if embedding_key in G.nodes[node]:
                return G.nodes[node][embedding_key].shape[0]
        raise ValueError(f"No node has '{embedding_key}' attribute. Cannot determine embedding dimension.")

    extra_embedding_dim = get_embedding_dim(G_extra, "features")

    X_extra = torch.tensor(
        np.array([
            G_extra.nodes[node].get("features", np.full(extra_embedding_dim, np.nan))
            for node in G_extra.nodes()
        ]),
        dtype=torch.float
    )

    df = pd.read_excel(data_path)
    y = torch.tensor(df['是否复发（1是 0否）'].values, dtype=torch.long)

    return X_extra,  extra_edge_index, y