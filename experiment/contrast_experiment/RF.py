import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, recall_score,
    precision_score, average_precision_score, confusion_matrix
)

# 设置全局随机种子
np.random.seed(42)

# 读取 CSV 文件
file_path = r'D:\PycharmProjects\Recurrence_prediction_0906\experiment\contrast_experiment\processed_data_modal_filled.csv'  # 替换为你的 CSV 文件路径
data = pd.read_csv(file_path)

# 分离特征和标签
X = data.iloc[:, :-1]  # 从第一列到倒数第二列是特征
y = data.iloc[:, -1]   # 最后一列是标签

# 初始化 StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 初始化评价指标列表
accuracy_scores = []
roc_auc_scores = []
f1_scores = []
recall_scores = []
precision_scores = []
aupr_scores = []

# 进行 5 折交叉验证
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(f"\n第 {fold + 1} 折的训练集和测试集索引：")
    print(f"训练集索引: {train_index}")
    print(f"测试集索引: {test_index}")

    # 划分训练集和测试集
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 从训练集中再分 20% 作为验证集（仅划分，不使用）
    train_idx, val_idx, _, _ = train_test_split(
        train_index, y_train, test_size=0.2, stratify=y_train, random_state=42)

    # 打印验证集索引
    print(f"验证集索引: {val_idx}")

    # 初始化随机森林模型
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 训练模型
    rf_model.fit(X_train, y_train)

    # 进行预测
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # 预测概率

    # 计算评价指标
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    aupr = average_precision_score(y_test, y_pred_proba)

    # 保存每一折的评价指标
    accuracy_scores.append(acc)
    roc_auc_scores.append(roc_auc)
    f1_scores.append(f1)
    recall_scores.append(recall)
    precision_scores.append(precision)
    aupr_scores.append(aupr)

    # 输出每一折的评价指标和混淆矩阵
    print(f"\n第 {fold + 1} 折的评价指标：")
    print(f"准确率 (ACC): {acc:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"F1 分数: {f1:.4f}")
    print(f"召回率 (RECALL): {recall:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print("混淆矩阵：")
    print(confusion_matrix(y_test, y_pred))

# 计算 5 折的平均评价指标和标准差
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
mean_roc_auc = np.mean(roc_auc_scores)
std_roc_auc = np.std(roc_auc_scores)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)
mean_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)
mean_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)
mean_aupr = np.mean(aupr_scores)
std_aupr = np.std(aupr_scores)

# 输出 5 折的平均评价指标和标准差
print("\n5 折交叉验证的平均评价指标：")
print(f"平均准确率 (ACC): {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"平均 AUC: {mean_roc_auc:.4f} ± {std_roc_auc:.4f}")
print(f"平均 F1 分数: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"平均召回率 (RECALL): {mean_recall:.4f} ± {std_recall:.4f}")
print(f"平均精确率 (Precision): {mean_precision:.4f} ± {std_precision:.4f}")
print(f"平均 AUPR: {mean_aupr:.4f} ± {std_aupr:.4f}")