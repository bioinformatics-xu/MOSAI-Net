import os
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from argparse import Namespace
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model.AMSF.mulit_GCN import Multi_GNN_model

def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def early_stopping(epoch, val_auc, best_val_auc, wait, patience, warm_up_epochs, model, best_model_state):
    if epoch < warm_up_epochs:
        return best_val_auc, wait, False, best_model_state

    if val_auc >= best_val_auc:
        best_val_auc = val_auc
        wait = 0
        best_model_state = model.state_dict()
    else:
        wait += 1
    if wait >= patience:
        return best_val_auc, wait, True, best_model_state
    return best_val_auc, wait, False, best_model_state

def main():
    seed = 42
    set_random_seed(seed)

    graph_dir = r"D:\PycharmProjects\Recurrence_prediction_0906\build_graph\graph_structure"
    data_path = r"D:\PycharmProjects\Recurrence_prediction_0906\build_graph\data\data_0906.xlsx"

    from model.data_loader.dataloader import load_data
    X_pet, X_pathology, X_gene, X_extra, pet_edge_index, pathology_edge_index, gene_edge_index, extra_edge_index, y = load_data(graph_dir, data_path)

    num_nodes = X_pet.size(0)
    num_classes = y.max().item() + 1

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_indices, test_indices) in enumerate(kf.split(range(num_nodes))):
        print(f"\n========== Fold {fold + 1}/5 ==========")

        train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=seed, stratify=y[train_indices].cpu().numpy())

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices] = True
        test_mask[test_indices] = True
        val_mask[val_indices] = True

        args = Namespace(
            num_layers=1,
            hidden_dim=64,
            dropout=0,
            conv_type="gcnmf",
            post_fusion_dropout=0,
        )

        model = Multi_GNN_model(
            pet_num_features=X_pet.size(1),
            pathology_num_features=X_pathology.size(1),
            gene_num_features=X_gene.size(1),
            extra_num_features=X_extra.size(1),
            num_classes=num_classes,
            pet_edge_index=pet_edge_index,
            pathology_edge_index=pathology_edge_index,
            gene_edge_index=gene_edge_index,
            extra_edge_index=extra_edge_index,
            pet_x=X_pet,
            pathology_x=X_pathology,
            gene_x=X_gene,
            extra_x=X_extra,
            args=args
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss()

        def train_step():
            model.train()
            optimizer.zero_grad()
            logits, _,_,_,_,_, _ = model(X_pet, X_pathology, X_gene, X_extra,
                                 pet_edge_index, pathology_edge_index, gene_edge_index, extra_edge_index)
            loss = criterion(logits[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()
            return loss.item()

        @torch.no_grad()
        def evaluate(mask):
            model.eval()
            logits, _,_,_,_, x_f, _ = model(X_pet, X_pathology, X_gene, X_extra,
                                 pet_edge_index, pathology_edge_index, gene_edge_index, extra_edge_index)
            probs = F.softmax(logits[mask], dim=1)
            preds = logits[mask].argmax(dim=1)
            acc = (preds == y[mask]).float().mean().item()

            if num_classes == 2:
                try:
                    auc = roc_auc_score(y[mask].cpu().numpy(), probs[:, 1].cpu().numpy())
                    aupr = average_precision_score(y[mask].cpu().numpy(), probs[:, 1].cpu().numpy())
                except ValueError as e:
                    print("AUC error:", e)
                    auc = float('nan')
                    aupr = float('nan')
            else:
                auc = float('nan')
                aupr = float('nan')

            f1 = f1_score(y[mask].cpu().numpy(), preds.cpu().numpy())
            recall = recall_score(y[mask].cpu().numpy(), preds.cpu().numpy())
            precision = precision_score(y[mask].cpu().numpy(), preds.cpu().numpy(), zero_division=0)

            return acc, preds, y[mask], auc, aupr, f1, recall, precision, x_f

        patience = 50
        warm_up_epochs = 50
        best_val_auc = 0
        wait = 0
        best_model_state = None

        for epoch in range(1, 1000):
            loss = train_step()

            train_acc, _, _, train_auc, train_aupr, train_f1, train_recall, train_precision, _ = evaluate(train_mask)
            val_acc, _, _, val_auc, val_aupr, val_f1, val_recall, val_precision, _ = evaluate(val_mask)

            best_val_auc, wait, stop_training, best_model_state = early_stopping(epoch, val_auc, best_val_auc, wait, patience, warm_up_epochs, model, best_model_state)

            print(f"Epoch {epoch:03d} | "
                  f"Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Train AUC: {train_auc:.4f} | Train AUPR: {train_aupr:.4f} | "
                  f"Train F1: {train_f1:.4f} | Train Recall: {train_recall:.4f} | Train Precision: {train_precision:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val AUPR: {val_aupr:.4f} | "
                  f"Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f} | Val Precision: {val_precision:.4f}")

            if stop_training:
                print(f"Early stopping at epoch {epoch}. Best Val AUC: {best_val_auc:.4f}")
                break

        model.load_state_dict(best_model_state)

        test_acc, test_pred, test_true, test_auc, test_aupr, test_f1, test_recall, test_precision, _ = evaluate(test_mask)
        print(f"Fold {fold + 1} Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}, Test AUPR: {test_aupr:.4f}, "
              f"Test F1: {test_f1:.4f}, Test Recall: {test_recall:.4f}, Test Precision: {test_precision:.4f}")

        fold_results.append({
            "fold": fold + 1,
            "test_acc": test_acc,
            "test_auc": test_auc,
            "test_aupr": test_aupr,
            "test_f1": test_f1,
            "test_recall": test_recall,
            "test_precision": test_precision,
            "confusion_matrix": confusion_matrix(test_true.cpu().numpy(), test_pred.cpu().numpy())
        })

    print("\n========== Cross-validation Summary ==========")
    for res in fold_results:
        print(f"Fold {res['fold']}: "
              f"Test Acc: {res['test_acc']:.4f}, Test AUC: {res['test_auc']:.4f}, Test AUPR: {res['test_aupr']:.4f}, "
              f"Test F1: {res['test_f1']:.4f}, Test Recall: {res['test_recall']:.4f}, Test Precision: {res['test_precision']:.4f}")
        print("Confusion Matrix:\n", res["confusion_matrix"])

    avg_acc = np.nanmean([r["test_acc"] for r in fold_results])
    avg_auc = np.nanmean([r["test_auc"] for r in fold_results])
    avg_aupr = np.nanmean([r["test_aupr"] for r in fold_results])
    avg_f1 = np.nanmean([r["test_f1"] for r in fold_results])
    avg_recall = np.nanmean([r["test_recall"] for r in fold_results])
    avg_precision = np.nanmean([r["test_precision"] for r in fold_results])

    std_acc = np.nanstd([r["test_acc"] for r in fold_results])
    std_auc = np.nanstd([r["test_auc"] for r in fold_results])
    std_aupr = np.nanstd([r["test_aupr"] for r in fold_results])
    std_f1 = np.nanstd([r["test_f1"] for r in fold_results])
    std_recall = np.nanstd([r["test_recall"] for r in fold_results])
    std_precision = np.nanstd([r["test_precision"] for r in fold_results])

    print(f"\nAverage Test Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Average Test AUC:      {avg_auc:.4f} ± {std_auc:.4f}")
    print(f"Average Test AUPR:     {avg_aupr:.4f} ± {std_aupr:.4f}")
    print(f"Average Test F1:       {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"Average Test Recall:   {avg_recall:.4f} ± {std_recall:.4f}")
    print(f"Average Test Precision:{avg_precision:.4f} ± {std_precision:.4f}")

if __name__ == "__main__":
    main()
