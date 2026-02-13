import numpy as np
import random
import torch
import argparse
import copy
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from model.mulit_graph_model import Mulit_Graph_model
from model.data_loader.dataloader import load_data

# ================ Parameter Settings ================
parser = argparse.ArgumentParser()
parser.add_argument('--r1', type=float, default=1, help='hyperparameter in loss function.')
parser.add_argument('--r2', type=float, default=0.5, help='hyperparameter in loss function.')
parser.add_argument('--r3', type=float, default=0, help='hyperparameter in loss function.')
parser.add_argument('--tau', type=float, default=1, help='hyperparameter in contrastive loss.')
parser.add_argument('--order', type=int, default=2, help='number of multi-hop graph.')
parser.add_argument('--nb_epochs', type=int, default=1000, help='maximal epochs.')
parser.add_argument('--patience', type=int, default=50, help='early stop.')
parser.add_argument('--nheads', type=int, default=8, help='number of heads in self-attention.')
parser.add_argument('--Trans_layer_num', type=int, default=2, help='layers number for self-attention.')
parser.add_argument('--lr', type=float, default=1e-3, help='learning ratio.')
parser.add_argument('--trans_dim', type=int, default=320, help='hidden dimension for transformer.')
parser.add_argument('--dropout_att', type=float, default=0.1, help='dropout in self-attention layers.')
parser.add_argument('--random_aug_feature', type=float, default=0.1, help='dropout in hidden layers.')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay.')
parser.add_argument('--warmup_num', type=int, default=50, help='epoch for warm-up.')
parser.add_argument('--ft_size', type=int, default=320, help='feature size of each node.')
parser.add_argument('--nclasses', type=int, default=2, help='number of classes in the dataset.')
# GCN parameters
parser.add_argument('--num_layers', type=int, default=1, help='')
parser.add_argument('--hidden_dim', type=int, default=64, help='')
parser.add_argument('--dropout', type=float, default=0, help='')
parser.add_argument('--conv_type', type=str, default="gcnmf", help='')
parser.add_argument('--jk', type=bool, default=True, help='whether to use jump connection')
# EMT
parser.add_argument('--post_fusion_dropout', type=float, default=0, help='')

args = parser.parse_args()

# ================ Main Function ================
if __name__ == '__main__':
    # Set random seed
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Set device
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    # Set paths
    graph_dir = r"build_graph/graph_structure"
    data_path = r"build_graph/data/data.xlsx"

    # Load data
    X_pet, X_pathology, X_gene, X_extra, pet_edge_index, pathology_edge_index, gene_edge_index, extra_edge_index, y = load_data(graph_dir, data_path)
    num_nodes = y.size(0)

    # Convert to numpy for sklearn processing
    y_np = y.numpy()

    # ================ 5-Fold Cross Validation ================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(num_nodes), y_np)):
        print(f"\n========== Fold {fold + 1}/5 ==========")

        # Split 20% from training set as validation set
        train_idx, val_idx, _, _ = train_test_split(
            train_idx, y_np[train_idx], test_size=0.2, stratify=y_np[train_idx], random_state=seed)

        # Convert to tensor
        idx_train = torch.tensor(train_idx, dtype=torch.long)
        idx_val = torch.tensor(val_idx, dtype=torch.long)
        idx_test = torch.tensor(test_idx, dtype=torch.long)

        # Initialize model
        embedder = Mulit_Graph_model(
            X_pet, X_pathology, X_gene, X_extra,
            pet_edge_index, pathology_edge_index, gene_edge_index, extra_edge_index,
            y, idx_train, idx_val, idx_test,
            copy.deepcopy(args), fold_id=fold
        )

        # Train
        training_time, stop_epoch, test_metrics = embedder.training()

        # Save metrics for each fold
        fold_metrics.append(test_metrics)

        # Print
        print(f"Fold {fold + 1} - ACC: {test_metrics['ACC']:.4f}, AUC: {test_metrics['AUC']:.4f}, "
              f"F1: {test_metrics['F1']:.4f}, RECALL: {test_metrics['RECALL']:.4f}, Precision: {test_metrics['Precision']:.4f}, AUPR: {test_metrics['AUPR']:.4f}")
        print(f"Fold {fold + 1} Confusion Matrix:\n{test_metrics['Confusion Matrix']}\n")

    # ================ Unified Repeated Output of Per-Fold Metrics ================
    print("\n========== Per-Fold Summary ==========")
    for fold, m in enumerate(fold_metrics, 1):
        print(f"Fold {fold} - "
              f"ACC: {m['ACC']:.4f}, AUC: {m['AUC']:.4f}, "
              f"F1: {m['F1']:.4f}, RECALL: {m['RECALL']:.4f}, Precision: {m['Precision']:.4f}, AUPR: {m['AUPR']:.4f}")
        print(f"Fold {fold} Confusion Matrix:\n{m['Confusion Matrix']}\n")

    # ================ Output Mean and Standard Deviation ================
    avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}

    print("\n========== 5-Fold Average ==========")
    print(f"Average ACC: {avg_metrics['ACC']:.4f} ± {std_metrics['ACC']:.4f}")
    print(f"Average AUC: {avg_metrics['AUC']:.4f} ± {std_metrics['AUC']:.4f}")
    print(f"Average F1:  {avg_metrics['F1']:.4f} ± {std_metrics['F1']:.4f}")
    print(f"Average RECALL: {avg_metrics['RECALL']:.4f} ± {std_metrics['RECALL']:.4f}")
    print(f"Average Precision: {avg_metrics['Precision']:.4f} ± {std_metrics['Precision']:.4f}")
    print(f"Average AUPR: {avg_metrics['AUPR']:.4f} ± {std_metrics['AUPR']:.4f}")