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
        modalities_to_build=["PET", "Pathology", "Gene", "Extra_Numeric"]  # New parameter to control which modality graphs to generate
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
    if "PET" in modalities_to_build:
        pet_embeddings, pet_indices = [], []
        for idx, text in df["PET/CT"].items():
            if pd.notna(text):
                pet_embeddings.append(bert_embed([str(text)])[0])
                pet_indices.append(idx)
        pet_embeddings = np.array(pet_embeddings)
        pet_emb_dict = dict(zip(pet_indices, pet_embeddings))  # Use original embeddings directly
        df["PET_Embedding"] = [pet_emb_dict.get(i, np.array([])) for i in df.index]

    # ---------------- 4. Pathology text embedding ----------------
    if "Pathology" in modalities_to_build:
        pathology_embeddings, pathology_indices = [], []
        for idx, text in df["Pathology"].items():
            if pd.notna(text):
                pathology_embeddings.append(bert_embed([str(text)])[0])
                pathology_indices.append(idx)
        pathology_embeddings = np.array(pathology_embeddings)
        pathology_emb_dict = dict(zip(pathology_indices, pathology_embeddings))  # Use original embeddings directly
        df["Pathology_Embedding"] = [pathology_emb_dict.get(i, np.array([])) for i in df.index]

    # ---------------- 5. Column definitions ----------------
    gene_columns = [col for col in df.columns if col not in [
        "Patient_ID", "Recurrence(1_Yes_0_No)", "Follow_up_Time", "Last_Medical_Record_Time", "Last_Chemo_Before_Recurrence_Time",
        "Recurrence_Time", "Time_To_Recurrence", "Notes", "PET/CT", "Pathology", "Pathology_Gross_Description",
        "PET_Embedding", "Pathology_Embedding",
        "Severe_Involvement_Other_Sites", "Age", "Gender", "Chemo_Cycles",
        "Stage", "Group(A1_B0)", "IPI(Only_Corrected)", "GCB(0_No_1_Yes)",
        "SUVmax(Stage/Prognosis:Global_Max)", "Mediastinal_Blood_Pool_SUVmax", "Liver_Right_Lobe_SUVmax", "Lesion_≥10cm(0_No_1_Yes)",
        "Bcl-2(0_Negative_1_Positive)", "CD10(0_Negative_1_Positive)", "BCL-6(0_Negative_1_Positive)", "c-MYC(0_Negative_1_Positive)",
        "P53(0_Negative_1_Positive)", "CD20(0_Negative_1_Positive)", "EBER(0_Negative_1_Positive)"
    ]]
    # Additional numeric columns
    EXTRA = ["SUVmax(Stage/Prognosis:Global_Max)", "Mediastinal_Blood_Pool_SUVmax", "Liver_Right_Lobe_SUVmax", "Lesion_≥10cm(0_No_1_Yes)", "Severe_Involvement_Other_Sites",
             "Bcl-2(0_Negative_1_Positive)", "CD10(0_Negative_1_Positive)", "BCL-6(0_Negative_1_Positive)", "c-MYC(0_Negative_1_Positive)",
             "Stage", "Group(A1_B0)", "IPI(Only_Corrected)", "GCB(0_No_1_Yes)"
             ]

    # ---------------- 6. Build graphs ----------------
    graphs = {}
    if "PET" in modalities_to_build:
        graphs["G_pet"] = nx.Graph()
        for idx in df.index:
            graphs["G_pet"].add_node(idx)
    if "Pathology" in modalities_to_build:
        graphs["G_pathology"] = nx.Graph()
        for idx in df.index:
            graphs["G_pathology"].add_node(idx)
    if "Gene" in modalities_to_build:
        graphs["G_gene"] = nx.Graph()
        for idx in df.index:
            graphs["G_gene"].add_node(idx)
    if "Extra_Numeric" in modalities_to_build:
        graphs["G_extra"] = nx.Graph()
        for idx in df.index:
            graphs["G_extra"].add_node(idx)

    # Node attributes
    for idx, row in df.iterrows():
        if "PET" in modalities_to_build and row["PET_Embedding"].size:
            graphs["G_pet"].nodes[idx]["features"] = row["PET_Embedding"].astype(float)
        if "Pathology" in modalities_to_build and row["Pathology_Embedding"].size:
            graphs["G_pathology"].nodes[idx]["features"] = row["Pathology_Embedding"].astype(float)
        if "Gene" in modalities_to_build and not row[gene_columns].isna().all():
            graphs["G_gene"].nodes[idx]["features"] = row[gene_columns].astype(float, errors='ignore').values
        if "Extra_Numeric" in modalities_to_build and not row[EXTRA].isna().all():
            graphs["G_extra"].nodes[idx]["features"] = row[EXTRA].astype(float, errors='ignore').values

    # ---------------- 7. PET graph ----------------
    if "PET" in modalities_to_build:
        pet_nodes = [(n, d) for n, d in graphs["G_pet"].nodes(data=True) if "features" in d]
        pet_idx = [n for n, _ in pet_nodes]
        pet_feat = [d["features"] for _, d in pet_nodes]
        if pet_feat:
            sim = pairwise_cosine_similarity_with_mask(np.array(pet_feat))
            for i in range(len(pet_idx)):
                s = sim[i]
                s[i] = -1  # Exclude self
                top = np.argsort(-s)[:top_k_connections]  # Get top top_k_connections nodes
                for j in top:
                    graphs["G_pet"].add_edge(pet_idx[i], pet_idx[j])

    # ---------------- 8. Pathology graph ----------------
    if "Pathology" in modalities_to_build:
        patho_nodes = [(n, d) for n, d in graphs["G_pathology"].nodes(data=True) if "features" in d]
        patho_idx = [n for n, _ in patho_nodes]
        patho_feat = [d["features"] for _, d in patho_nodes]
        if patho_feat:
            sim = pairwise_cosine_similarity_with_mask(np.array(patho_feat))
            for i in range(len(patho_idx)):
                s = sim[i]
                s[i] = -1  # Exclude self
                top = np.argsort(-s)[:top_k_connections]  # Get top top_k_connections nodes
                for j in top:
                    graphs["G_pathology"].add_edge(patho_idx[i], patho_idx[j])

    # ---------------- 9. Gene graph ----------------
    if "Gene" in modalities_to_build:
        gene_nodes = [(n, d) for n, d in graphs["G_gene"].nodes(data=True) if "features" in d]
        gene_idx = [n for n, _ in gene_nodes]
        gene_feat = [d["features"] for _, d in gene_nodes]
        if gene_feat:
            dist = pairwise_euclidean_distance_with_mask(np.array(gene_feat))
            for i in range(len(gene_idx)):
                d = dist[i]
                d[i] = np.inf  # Exclude self
                top = np.argsort(d)[:top_k_connections]  # Get top top_k_connections nodes
                for j in top:
                    graphs["G_gene"].add_edge(gene_idx[i], gene_idx[j])

    # ---------------- 10. Extra numeric columns graph ----------------
    if "Extra_Numeric" in modalities_to_build:
        extra_nodes = [(n, d) for n, d in graphs["G_extra"].nodes(data=True) if "features" in d]
        extra_idx = [n for n, _ in extra_nodes]
        extra_feat = [d["features"] for _, d in extra_nodes]
        dist = pairwise_euclidean_distance_with_mask(np.array(extra_feat))
        for i in range(len(extra_idx)):
            d = dist[i]
            d[i] = np.inf  # Exclude self
            top = np.argsort(d)[:top_k_connections]  # Get top top_k_connections nodes
            for j in top:
                graphs["G_extra"].add_edge(extra_idx[i], extra_idx[j])

    # ---------------- 11. Handle empty nodes ----------------
    def handle_empty_nodes(G, G1, G2, G3):
        empty = [n for n in G.nodes if "features" not in G.nodes[n]]
        for n in empty:
            neighbors1 = set(G1.neighbors(n)) if n in G1 else set()
            neighbors2 = set(G2.neighbors(n)) if n in G2 else set()
            neighbors3 = set(G3.neighbors(n)) if n in G3 else set()
            neighbors = neighbors1.union(neighbors2).union(neighbors3)
            for neighbor in neighbors:
                G.add_edge(n, neighbor)

    if "PET" in modalities_to_build:
        G_pathology = graphs.get("G_pathology", nx.Graph()) if "Pathology" in modalities_to_build else nx.Graph()
        G_gene = graphs.get("G_gene", nx.Graph()) if "Gene" in modalities_to_build else nx.Graph()
        G_extra = graphs.get("G_extra", nx.Graph()) if "Extra_Numeric" in modalities_to_build else nx.Graph()
        handle_empty_nodes(graphs["G_pet"], G_pathology, G_gene, G_extra)

    if "Pathology" in modalities_to_build:
        G_pet = graphs.get("G_pet", nx.Graph()) if "PET" in modalities_to_build else nx.Graph()
        G_gene = graphs.get("G_gene", nx.Graph()) if "Gene" in modalities_to_build else nx.Graph()
        G_extra = graphs.get("G_extra", nx.Graph()) if "Extra_Numeric" in modalities_to_build else nx.Graph()
        handle_empty_nodes(graphs["G_pathology"], G_pet, G_gene, G_extra)

    if "Gene" in modalities_to_build:
        G_pet = graphs.get("G_pet", nx.Graph()) if "PET" in modalities_to_build else nx.Graph()
        G_pathology = graphs.get("G_pathology", nx.Graph()) if "Pathology" in modalities_to_build else nx.Graph()
        G_extra = graphs.get("G_extra", nx.Graph()) if "Extra_Numeric" in modalities_to_build else nx.Graph()
        handle_empty_nodes(graphs["G_gene"], G_pet, G_pathology, G_extra)

    if "Extra_Numeric" in modalities_to_build:
        G_pet = graphs.get("G_pet", nx.Graph()) if "PET" in modalities_to_build else nx.Graph()
        G_pathology = graphs.get("G_pathology", nx.Graph()) if "Pathology" in modalities_to_build else nx.Graph()
        G_gene = graphs.get("G_gene", nx.Graph()) if "Gene" in modalities_to_build else nx.Graph()
        handle_empty_nodes(graphs["G_extra"], G_pet, G_pathology, G_gene)

    # ---------------- 12. Self-loops ----------------
    for G in graphs.values():
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

    for G in graphs.values():
        jaccards_coefficient(G)

    # ---------------- 14. Visualization ----------------
    def plot_adj(G, title, path):
        plt.figure(figsize=(10, 8))
        plt.imshow(nx.to_numpy_array(G), cmap='Blues', interpolation='nearest')
        plt.title(title)
        plt.colorbar()
        plt.savefig(path)
        plt.close()

    if "PET" in modalities_to_build:
        plot_adj(graphs["G_pet"], "PET Graph Adjacency Matrix", f"{output_dir}/PET_adjacency_matrix.png")
    if "Pathology" in modalities_to_build:
        plot_adj(graphs["G_pathology"], "Pathology Graph Adjacency Matrix", f"{output_dir}/Pathology_adjacency_matrix.png")
    if "Gene" in modalities_to_build:
        plot_adj(graphs["G_gene"], "Gene Graph Adjacency Matrix", f"{output_dir}/Gene_adjacency_matrix.png")
    if "Extra_Numeric" in modalities_to_build:
        plot_adj(graphs["G_extra"], "Extra Numeric Columns Graph Adjacency Matrix", f"{output_dir}/Extra_numeric_adjacency_matrix.png")

    # ---------------- 15. Save ----------------
    for name, G in graphs.items():
        with open(f"{output_dir}/{name}.pkl", "wb") as f:
            pickle.dump(G, f)

    return graphs

# ---------------- 16. Entry point ----------------
if __name__ == "__main__":
    create_graphs(
        data_path=r"data/data.xlsx",
        top_k_connections=7,
        output_dir=r"graph_structure",
        #modalities_to_build=["PET", "Pathology", "Gene", "Extra_Numeric"]
        modalities_to_build=["PET", "Pathology", "Extra_Numeric"]
    )