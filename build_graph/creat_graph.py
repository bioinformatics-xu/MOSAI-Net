import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel

# Custom distance / similarity calculation (must exist)
from weighted_similarity import pairwise_cosine_similarity_with_mask
from weighted_distance import pairwise_euclidean_distance_with_mask

# Chinese display
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.unicode_minus'] = False


def create_graphs(
        data_path,
        top_k_connections,
        output_dir,
):
    # ---------------- 1. Load data ----------------
    df = pd.read_excel(data_path)

    # ---------------- 2. Load BERT ----------------
    model_path = r'bert_cn'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertModel.from_pretrained(model_path)

    @torch.no_grad()
    def bert_embed(texts):
        if not texts:
            return [np.array([]) for _ in range(len(texts))]
        encoded = tokenizer(texts, padding=True, truncation=True,
                            max_length=512, return_tensors='pt')
        cls_vec = bert_model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()
        return [vec for vec in cls_vec]

    # ---------------- 3. PET text embedding ----------------
    pet_embeddings, pet_indices = [], []
    for idx, text in df["PET/CT"].items():
        if pd.notna(text):
            pet_embeddings.append(bert_embed([str(text)])[0])
            pet_indices.append(idx)
    pet_embeddings = np.array(pet_embeddings)
    pet_emb_dict = dict(zip(pet_indices, pet_embeddings))  # Use original embeddings directly
    df["PET嵌入"] = [pet_emb_dict.get(i, np.array([])) for i in df.index]

    # ---------------- 4. Pathology text embedding ----------------
    pathology_embeddings, pathology_indices = [], []
    for idx, text in df["病理"].items():
        if pd.notna(text):
            pathology_embeddings.append(bert_embed([str(text)])[0])
            pathology_indices.append(idx)
    pathology_embeddings = np.array(pathology_embeddings)
    pathology_emb_dict = dict(zip(pathology_indices, pathology_embeddings))  # Use original embeddings directly
    df["病理嵌入"] = [pathology_emb_dict.get(i, np.array([])) for i in df.index]

    # ---------------- 5. Column definitions ----------------
    gene_columns = [col for col in df.columns if col not in [
        "患者ID", "是否复发（1是 0否）", "随访时间", "最后一次病历时间", "复发前末次化疗时间",
        "复发时间", "多久时间内复发/未复发", "备注", "PET/CT", "病理", "病理肉眼所见",
        "PET嵌入", "病理嵌入",
        "是否严重累及其他部位","年龄","性别","化疗次数",
        "分期","分组（A1 B0）", "IPI（只会修正不会变化）", "GCB（0否 1是）",
        "SUVmax（分期/预后：取全身最高）", "纵隔血池SUVmax", "肝右叶SUVmax","存在 ≥10 cm 病灶（三维最大径 ≥10 cm）0否1是",
        "Bcl-2（0阴性1阳性）","CD10（0阴性1阳性）", "BCL-6（0阴性1阳性）", "c-MYC（0阴性1阳性）",
        "P53（0阴性1阳性）","CD20（0阴性1阳性）","EBER（0阴性1阳性）"
    ]]
    # Additional numeric columns
    EXTRA = ["SUVmax（分期/预后：取全身最高）","纵隔血池SUVmax", "肝右叶SUVmax", "存在 ≥10 cm 病灶（三维最大径 ≥10 cm）0否1是", "是否严重累及其他部位",
             "Bcl-2（0阴性1阳性）","CD10（0阴性1阳性）","BCL-6（0阴性1阳性）", "c-MYC（0阴性1阳性）",
             "分期","分组（A1 B0）","IPI（只会修正不会变化）", "GCB（0否 1是）"
             ]

    # ---------------- 6. Build graphs ----------------
    G_pet, G_pathology, G_gene, G_extra = nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()
    for idx in df.index:
        G_pet.add_node(idx)
        G_pathology.add_node(idx)
        G_gene.add_node(idx)
        G_extra.add_node(idx)

    # Node attributes
    for idx, row in df.iterrows():
        if row["PET嵌入"].size:
            G_pet.nodes[idx]["features"] = row["PET嵌入"].astype(float)
        if row["病理嵌入"].size:
            G_pathology.nodes[idx]["features"] = row["病理嵌入"].astype(float)
        if not row[gene_columns].isna().all():
            G_gene.nodes[idx]["features"] = row[gene_columns].astype(float, errors='ignore').values
        if not row[EXTRA].isna().all():
            G_extra.nodes[idx]["features"] = row[EXTRA].astype(float, errors='ignore').values

    # ---------------- 7. PET graph ----------------
    pet_nodes = [(n, d) for n, d in G_pet.nodes(data=True) if "features" in d]
    pet_idx = [n for n, _ in pet_nodes]
    pet_feat = [d["features"] for _, d in pet_nodes]
    if pet_feat:
        sim = pairwise_cosine_similarity_with_mask(np.array(pet_feat))
        for i in range(len(pet_idx)):
            s = sim[i]
            s[i] = -1  # Exclude self
            top = np.argsort(-s)[:top_k_connections]  # Get top top_k_connections nodes
            for j in top:
                G_pet.add_edge(pet_idx[i], pet_idx[j])

    # ---------------- 8. Pathology graph ----------------
    patho_nodes = [(n, d) for n, d in G_pathology.nodes(data=True) if "features" in d]
    patho_idx = [n for n, _ in patho_nodes]
    patho_feat = [d["features"] for _, d in patho_nodes]
    if patho_feat:
        sim = pairwise_cosine_similarity_with_mask(np.array(patho_feat))
        for i in range(len(patho_idx)):
            s = sim[i]
            s[i] = -1  # Exclude self
            top = np.argsort(-s)[:top_k_connections]  # Get top top_k_connections nodes
            for j in top:
                G_pathology.add_edge(patho_idx[i], patho_idx[j])

        # ---------------- 9. Gene graph ----------------
        gene_nodes = [(n, d) for n, d in G_gene.nodes(data=True) if "features" in d]
        gene_idx = [n for n, _ in gene_nodes]
        gene_feat = [d["features"] for _, d in gene_nodes]
        if gene_feat:
            dist = pairwise_euclidean_distance_with_mask(np.array(gene_feat))
            for i in range(len(gene_idx)):
                d = dist[i]
                d[i] = np.inf  # Exclude self
                top = np.argsort(d)[:top_k_connections]  # Get top top_k_connections nodes
                for j in top:
                    G_gene.add_edge(gene_idx[i], gene_idx[j])

    # ---------------- 10. Extra numeric columns graph ----------------
    # Extract nodes with "features" attribute and their features
    extra_nodes = [(n, d) for n, d in G_extra.nodes(data=True) if "features" in d]
    extra_idx = [n for n, _ in extra_nodes]
    extra_feat = [d["features"] for _, d in extra_nodes]
    dist = pairwise_euclidean_distance_with_mask(np.array(extra_feat))
    for i in range(len(extra_idx)):
        d = dist[i]
        d[i] = np.inf  # Exclude self
        top = np.argsort(d)[:top_k_connections]  # Get top top_k_connections nodes
        for j in top:
            G_extra.add_edge(extra_idx[i], extra_idx[j])

    # ---------------- 11. Handle empty nodes ----------------
    def handle_empty_nodes(G, G1, G2, G3):
        empty = [n for n in G.nodes if "features" not in G.nodes[n]]
        for n in empty:
            neighbors1 = set(G1.neighbors(n))
            neighbors2 = set(G2.neighbors(n))
            neighbors3 = set(G3.neighbors(n))
            neighbors = neighbors1.union(neighbors2).union(neighbors3)
            for neighbor in neighbors:
                G.add_edge(n, neighbor)

    handle_empty_nodes(G_pet, G_pathology, G_gene, G_extra)
    handle_empty_nodes(G_pathology, G_pet, G_gene, G_extra)
    handle_empty_nodes(G_gene, G_pet, G_pathology, G_extra)
    handle_empty_nodes(G_extra, G_pet, G_pathology, G_gene)

    # ---------------- 12. Self-loops ----------------
    for G in (G_pet, G_pathology, G_gene, G_extra):
        for n in G.nodes:
            if not G.has_edge(n, n):
                G.add_edge(n, n)

    # ---------------- 13. Apply Jaccard's Coefficient to remove noisy edges ----------------
    def jaccards_coefficient(G):
        edges_to_remove = []
        for u, v in G.edges():
            neighbors_u = set(G.neighbors(u))
            neighbors_v = set(G.neighbors(v))
            common_neighbors = neighbors_u.intersection(neighbors_v)
            all_neighbors = neighbors_u.union(neighbors_v)
            if len(all_neighbors) == 0:
                jaccard = 0
            else:
                jaccard = len(common_neighbors) / len(all_neighbors)
            if jaccard < 0.1:  # Set threshold to 0.1, can be adjusted as needed
                edges_to_remove.append((u, v))
        for u, v in edges_to_remove:
            G.remove_edge(u, v)

    # Apply Jaccard's Coefficient to remove noisy edges for each modality graph
    jaccards_coefficient(G_pet)
    jaccards_coefficient(G_pathology)
    jaccards_coefficient(G_gene)
    jaccards_coefficient(G_extra)

    # ---------------- 13. Visualization ----------------
    def plot_adj(G, title, path):
        plt.figure(figsize=(10, 8))
        plt.imshow(nx.to_numpy_array(G), cmap='Blues', interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        plt.savefig(path)
        plt.close()

    plot_adj(G_pet, "PET Graph Adjacency Matrix", f"{output_dir}/PET_adjacency_matrix.png")
    plot_adj(G_pathology, "Pathology Graph Adjacency Matrix", f"{output_dir}/Pathology_adjacency_matrix.png")
    plot_adj(G_gene, "Gene Graph Adjacency Matrix", f"{output_dir}/Gene_adjacency_matrix.png")
    plot_adj(G_extra, "Extra Numeric Columns Graph Adjacency Matrix", f"{output_dir}/Extra_numeric_adjacency_matrix.png")

    # ---------------- 14. Save ----------------
    for name, G in zip(["G_pet", "G_pathology", "G_gene", "G_extra"],
                       [G_pet, G_pathology, G_gene, G_extra]):
        with open(f"{output_dir}/{name}.pkl", "wb") as f:
            pickle.dump(G, f)

    return G_pet, G_pathology, G_gene, G_extra

# ---------------- 15. Entry point ----------------
if __name__ == "__main__":
    create_graphs(
        data_path=r"data/data.xlsx",
        top_k_connections=7,
        output_dir=r"graph_structure",
    )