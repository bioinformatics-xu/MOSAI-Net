import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score,
    precision_score, average_precision_score, confusion_matrix
)

# 读取CSV文件
file_path = r'D:\PycharmProjects\Recurrence_prediction_0906\experiment\contrast_experiment\processed_data_modal_filled.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 分离特征和标签
X = df.drop(columns=['label']).values
y = df['label'].values

# 5折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 初始化评价指标列表
accuracies = []
roc_aucs = []
f1_scores = []
recalls = []
precisions = []
auprs = []

for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}")

    # 打印训练集和测试集的索引
    print(f"  Training indices: {train_index}")
    print(f"  Testing indices: {test_index}")

    # 分层采样划分训练集和测试集
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # 将训练集再分出20%作为验证集，同时保证分层采样
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # 数据标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 创建MLP分类器
    mlp = MLPClassifier(hidden_layer_sizes=(1000, 500, 250), max_iter=500, activation='relu', solver='adam', random_state=42)

    # 训练模型
    mlp.fit(X_train, y_train)

    # 预测测试集
    y_pred = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)[:, 1]  # 获取正类的概率

    # 计算评价指标
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    aupr = average_precision_score(y_test, y_pred_proba)

    # 保存每一折的评价指标
    accuracies.append(accuracy)
    roc_aucs.append(roc_auc)
    f1_scores.append(f1)
    recalls.append(recall)
    precisions.append(precision)
    auprs.append(aupr)

    print(f"  Evaluation metrics for fold {fold + 1}:")
    print(f"    Accuracy (ACC): {accuracy:.4f}")
    print(f"    ROC-AUC (AUC): {roc_auc:.4f}")
    print(f"    F1 Score (F1): {f1:.4f}")
    print(f"    Recall (Recall): {recall:.4f}")
    print(f"    Precision (Precision): {precision:.4f}")
    print(f"    AUPR (AUPR): {aupr:.4f}")
    print()

# 打印平均评价指标和标准差
print("Average evaluation metrics across all folds:")
print(f"  Mean Accuracy (ACC): {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"  Mean ROC-AUC (AUC): {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")
print(f"  Mean F1 Score (F1): {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"  Mean Recall (Recall): {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"  Mean Precision (Precision): {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"  Mean AUPR (AUPR): {np.mean(auprs):.4f} ± {np.std(auprs):.4f}")