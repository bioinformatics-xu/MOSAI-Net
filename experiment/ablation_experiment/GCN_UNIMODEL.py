import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from argparse import Namespace
import pandas as pd
import numpy as np
import random

from model.data_loader.dataloader import load_data
from model.GNN_model.models import get_model
from torch_geometric.utils import to_undirected

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
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

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        pred_prob = torch.exp(out[mask])
        pred = pred_prob.max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        try:
            auc = roc_auc_score(data.y[mask].cpu().numpy(), pred_prob[:, 1].cpu().numpy())
        except ValueError as e:
            print(f"AUC calculation error: {e}")
            auc = float('nan')

    return acc, pred, auc

def main():
    seed = 42
    set_random_seed(seed)

    graph_dir = r"/build_graph/graph_structure"
    data_path = r"/build_graph/data/data.xlsx"

    X_pet, X_pathology, X_gene, X_extra, pet_edge_index, pathology_edge_index, gene_edge_index, extra_edge_index, y = load_data(graph_dir, data_path)

    x = X_extra
    edge_index = extra_edge_index
    num_nodes = x.size(0)
    num_features = x.size(1)
    num_classes = y.max().item() + 1

    data = Data(x=x, edge_index=edge_index, y=y)

    args = Namespace(
        num_layers=1,
        hidden_dim=64,
        dropout=0,
        conv_type="gcnmf"
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []

    for fold, (train_indices, test_indices) in enumerate(skf.split(data.x, y)):
        print(f"Fold {fold + 1}/5")

        train_indices, val_indices, _, _ = train_test_split(
            train_indices, y[train_indices], test_size=0.2, stratify=y[train_indices], random_state=seed)

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_indices] = True
        data.val_mask[val_indices] = True
        data.test_mask[test_indices] = True

        model = get_model("gcnmf", num_features, num_classes, edge_index, x, args)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.NLLLoss()

        patience = 100
        warm_up_epochs = 50
        best_val_auc = 0
        wait = 0
        best_model_state = None
        for epoch in range(1, 1000):
            loss = train(model, data, optimizer, criterion)
            train_acc, _, train_auc = evaluate(model, data, data.train_mask)
            val_acc, _, val_auc = evaluate(model, data, data.val_mask)
            print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Train AUC: {train_auc:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

            best_val_auc, wait, stop_training, best_model_state = early_stopping(epoch, val_auc, best_val_auc, wait, patience, warm_up_epochs, model,best_model_state)

            if stop_training:
                print(f"Early stopping at epoch {epoch}. Best Val AUC: {best_val_auc:.4f}")
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_acc, test_pred, test_auc = evaluate(model, data, data.test_mask)
        print(f"Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

        true_labels = data.y[data.test_mask]
        cm = confusion_matrix(true_labels.cpu().numpy(), test_pred.cpu().numpy())
        print("Confusion Matrix (GNN):")
        print(cm)

        X_train = x[train_indices].cpu().numpy()
        y_train = y[train_indices].cpu().numpy()
        X_test = x[test_indices].cpu().numpy()
        y_test = y[test_indices].cpu().numpy()

        rf_model = RandomForestClassifier(n_estimators=100, random_state=seed)
        rf_model.fit(X_train, y_train)

        feature_importances = rf_model.feature_importances_

        print("Feature Importances (Random Forest):")
        for i, importance in enumerate(feature_importances):
            print(f"Feature {i}: Importance = {importance:.4f}")

        y_pred = rf_model.predict(X_test)
        rf_test_acc = accuracy_score(y_test, y_pred)
        rf_test_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

        print(f"Random Forest Test Acc: {rf_test_acc:.4f}, Random Forest Test AUC: {rf_test_auc:.4f}")

        rf_cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix (Random Forest):")
        print(rf_cm)

        fold_results.append({
            'fold': fold + 1,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'confusion_matrix': cm,
            'rf_test_acc': rf_test_acc,
            'rf_test_auc': rf_test_auc,
            'rf_confusion_matrix': rf_cm
        })

    for result in fold_results:
        print(f"Fold {result['fold']}:")
        print(f"  GNN Test Acc: {result['test_acc']:.4f}, GNN Test AUC: {result['test_auc']:.4f}")
        print("  GNN Confusion Matrix:")
        print(result['confusion_matrix'])
        print(f"  Random Forest Test Acc: {result['rf_test_acc']:.4f}, Random Forest Test AUC: {result['rf_test_auc']:.4f}")
        print("  Random Forest Confusion Matrix:")
        print(result['rf_confusion_matrix'])

    avg_gnn_test_acc = np.mean([result['test_acc'] for result in fold_results])
    avg_gnn_test_auc = np.mean([result['test_auc'] for result in fold_results])
    avg_rf_test_acc = np.mean([result['rf_test_acc'] for result in fold_results])
    avg_rf_test_auc = np.mean([result['rf_test_auc'] for result in fold_results])

    print("\nAverage Results Over 5 Folds:")
    print(f"  Average GNN Test Accuracy: {avg_gnn_test_acc:.4f}")
    print(f"  Average GNN Test AUC: {avg_gnn_test_auc:.4f}")
    print(f"  Average Random Forest Test Accuracy: {avg_rf_test_acc:.4f}")
    print(f"  Average Random Forest Test AUC: {avg_rf_test_auc:.4f}")

if __name__ == "__main__":
    main()