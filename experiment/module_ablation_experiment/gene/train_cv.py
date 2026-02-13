import numpy as np
import random
import torch
import argparse
import copy
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from mulit_graph_model import Mulit_Graph_model
from dataloader import load_data

# ================ 参数设置 ================
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
parser.add_argument('--lr', type=float, default=0.001, help='learning ratio.')
parser.add_argument('--trans_dim', type=int, default=64, help='hidden dimension for transformer.')
parser.add_argument('--dropout_att', type=float, default=0.1, help='dropout in self-attention layers.')
parser.add_argument('--random_aug_feature', type=float, default=0.1, help='dropout in hidden layers.')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay.')
parser.add_argument('--warmup_num', type=int, default=50, help='epoch for warm-up.')
parser.add_argument('--ft_size', type=int, default=64, help='feature size of each node.')
parser.add_argument('--nclasses', type=int, default=2, help='number of classes in the dataset.')
# GCN 参数
parser.add_argument('--num_layers', type=int, default=1, help='')
parser.add_argument('--hidden_dim', type=int, default=64, help='')
parser.add_argument('--dropout', type=float, default=0, help='')
parser.add_argument('--conv_type', type=str, default="gcnmf", help='')
parser.add_argument('--jk', type=bool, default=True, help='是否使用跳跃连接')
# EMT
parser.add_argument('--post_fusion_dropout', type=float, default=0, help='')

args = parser.parse_args()

# ================ 主函数 ================
if __name__ == '__main__':
    # 设置随机种子
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # 设置设备
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
    else:
        args.device = torch.device('cpu')

    # 设置路径
    graph_dir = r"D:\PycharmProjects\Recurrence_prediction_0906\experiment\module_ablation_experiment\gene\graph_structure"
    data_path = r"D:\PycharmProjects\Recurrence_prediction_0906\build_graph\data\data_0906.xlsx"

    # 加载数据
    X_gene, gene_edge_index, y = load_data(graph_dir, data_path)
    num_nodes = y.size(0)

    # 转 numpy 方便 sklearn 处理
    y_np = y.numpy()

    # ================ 5折交叉验证 ================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(num_nodes), y_np)):
        print(f"\n========== Fold {fold + 1}/5 ==========")

        # 从训练集中再分 20% 作为验证集
        train_idx, val_idx, _, _ = train_test_split(
            train_idx, y_np[train_idx], test_size=0.2, stratify=y_np[train_idx], random_state=seed)

        # 转 tensor
        idx_train = torch.tensor(train_idx, dtype=torch.long)
        idx_val = torch.tensor(val_idx, dtype=torch.long)
        idx_test = torch.tensor(test_idx, dtype=torch.long)

        # 初始化模型
        embedder = Mulit_Graph_model(
            X_gene,
            gene_edge_index,
            y, idx_train, idx_val, idx_test,
            copy.deepcopy(args)
        )

        # 训练
        training_time, stop_epoch, test_metrics = embedder.training()

        # 保存每折指标
        fold_metrics.append(test_metrics)

        # 打印
        print(f"Fold {fold + 1} - ACC: {test_metrics['ACC']:.4f}, AUC: {test_metrics['AUC']:.4f}, "
              f"F1: {test_metrics['F1']:.4f}, RECALL: {test_metrics['RECALL']:.4f}, Precision: {test_metrics['Precision']:.4f}, AUPR: {test_metrics['AUPR']:.4f}")
        print(f"Fold {fold + 1} Confusion Matrix:\n{test_metrics['Confusion Matrix']}\n")

    # ================ 统一重复输出每折指标 ================
    print("\n========== Per-Fold Summary ==========")
    for fold, m in enumerate(fold_metrics, 1):
        print(f"Fold {fold} - "
              f"ACC: {m['ACC']:.4f}, AUC: {m['AUC']:.4f}, "
              f"F1: {m['F1']:.4f}, RECALL: {m['RECALL']:.4f}, Precision: {m['Precision']:.4f}, AUPR: {m['AUPR']:.4f}")
        print(f"Fold {fold} Confusion Matrix:\n{m['Confusion Matrix']}\n")

    # ================ 输出平均值和标准差 ================
    avg_metrics = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
    std_metrics = {k: np.std([m[k] for m in fold_metrics]) for k in fold_metrics[0]}

    print("\n========== 5-Fold Average ==========")
    print(f"Average ACC: {avg_metrics['ACC']:.4f} ± {std_metrics['ACC']:.4f}")
    print(f"Average AUC: {avg_metrics['AUC']:.4f} ± {std_metrics['AUC']:.4f}")
    print(f"Average F1:  {avg_metrics['F1']:.4f} ± {std_metrics['F1']:.4f}")
    print(f"Average RECALL: {avg_metrics['RECALL']:.4f} ± {std_metrics['RECALL']:.4f}")
    print(f"Average Precision: {avg_metrics['Precision']:.4f} ± {std_metrics['Precision']:.4f}")
    print(f"Average AUPR: {avg_metrics['AUPR']:.4f} ± {std_metrics['AUPR']:.4f}")