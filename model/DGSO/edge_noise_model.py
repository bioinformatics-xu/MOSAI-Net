import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix, f1_score, \
    balanced_accuracy_score, precision_score, average_precision_score


def fix_isolated_nodes_dense(adj_dense, sim_matrix):
    """
    adj_dense: [N, N] dense adjacency matrix (0/1)
    sim_matrix: [N, N] node similarity matrix (diagonal excluded)
    Returns: repaired adj_dense (modified in-place)
    """
    degree = adj_dense.sum(1)  # [N]
    isolated = (degree == 0)   # bool [N]
    if isolated.sum() == 0:
        return adj_dense

    # For each isolated node, find the most similar non-self node
    for idx in torch.where(isolated)[0]:
        best_j = sim_matrix[idx].argmax()
        adj_dense[idx, best_j] = 1.0
        adj_dense[best_j, idx] = 1.0  # Undirected graph symmetry
    return adj_dense

# ---------- 1. Calculate monitoring metrics ----------
def monitor_score(metrics):
    return metrics['AUC'] # + metrics['ACC'] #+ metrics['F1'] + metrics['RECALL']

def evaluate_metrics(y_true, y_pred, y_prob):
    """
    Binary classification specific: ACC, AUC, F1, RECALL, Precision, AUPR, confusion matrix (zero warning).

    Parameters:
        y_true (np.ndarray): True labels (0/1).
        y_pred (np.ndarray): Predicted classes (0/1).
        y_prob (np.ndarray): Positive class probabilities (shape=(n_samples,) or (n_samples, 2)).

    Returns:
        dict: Dictionary of metrics.
    """
    # Use positive class probability uniformly
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    # Basic metrics (zero_division=0 eliminates warnings)
    acc   = accuracy_score(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_prob)
    f1    = f1_score(y_true, y_pred)
    recall= recall_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred,zero_division=0)
    aupr  = average_precision_score(y_true, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        'ACC': acc,
        'AUC': auc,
        'F1': f1,
        'RECALL': recall,
        'Precision': prec,
        'AUPR': aupr,
        'Confusion Matrix': cm
    }

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def row_normalize(A):
    """Row-normalize dense matrix"""
    eps = 2.2204e-16
    rowsum = A.sum(dim=-1).clamp(min=0.) + eps
    r_inv = rowsum.pow(-1)
    A = r_inv.unsqueeze(-1) * A
    return A


def robust_contrastive_loss(adj_matrix, sim_matrix, pred_prob, temperature=0.5):
    """
    Corrected robust contrastive loss function, only pulls positive sample pairs closer
    adj_matrix: Adjacency matrix [N, N]
    sim_matrix: Node similarity matrix [N, N]
    pred_prob: Model predicted class probabilities [N, C]
    temperature: Contrastive loss temperature coefficient
    """
    N = adj_matrix.size(0)

    # 1. Build base positive sample mask
    pos_mask = adj_matrix > 0  # Original positive samples (node pairs with edge connections)
    pos_mask.fill_diagonal_(False)  # Exclude self

    # 2. Calculate class similarity matrix (cosine similarity)
    # Normalize prediction probability vectors
    #pred_prob_norm = pred_prob / pred_prob.norm(dim=1, keepdim=True)
    class_sim_matrix = torch.mm(pred_prob, pred_prob.t())  # [N, N]

    # 4. Dynamically adjust sample reliability ---------------------------------------------------------
    reliable_pos = pos_mask & (class_sim_matrix > 0.65)  # Dynamic positive samples: connected by edge and predicted class similarity

    # 5. Calculate weighted contrastive loss ----------------------------------------------------------
    losses = []
    for i in range(N):
        # Positive sample part (pull reliable positive samples closer)
        pos_indices = reliable_pos[i].nonzero(as_tuple=False).squeeze(1)
        if len(pos_indices) > 0:
            pos_sim = sim_matrix[i, pos_indices]
            # Use torch.sigmoid to normalize similarity values
            pos_sim = torch.sigmoid(pos_sim / temperature)
            pos_loss = -torch.log(pos_sim + 1e-8).mean()
        else:
            pos_loss = torch.tensor(0.0, device=adj_matrix.device)

        losses.append(pos_loss)

    return torch.mean(torch.stack(losses)) if losses else torch.tensor(0.0, device=adj_matrix.device)

def Ncontrast(x_dis, adj_label, tau = 1, train_index_sort=None):
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum_mid = torch.sum(x_dis, 1)
    x_dis_sum_pos_mid = torch.sum(x_dis*adj_label, 1)
    x_dis_sum = x_dis_sum_mid[train_index_sort]
    x_dis_sum_pos = x_dis_sum_pos_mid[train_index_sort]
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum**(-1))+1e-8).mean()
    return loss

def get_A_r(adj, r):
    adj_label = adj
    for i in range(r - 1):
        adj_label = adj_label @ adj
    return adj_label



class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EfficientAttention(nn.Module):
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Linear(in_channels, key_channels)
        self.queries = nn.Linear(in_channels, key_channels)
        self.values = nn.Linear(in_channels, value_channels)
        self.reprojection = nn.Linear(key_channels, key_channels)

    def forward(self, input_):
        keys = self.keys(input_)
        queries = self.queries(input_)
        values = self.values(input_)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        attention_scores_list = []  # For storing attention score matrix for each head
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels], dim=0)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels]
            context = key.transpose(0, 1) @ value
            attended_value = query @ context
            attended_values.append(attended_value)

            # Calculate attention score matrix for current head
            attention_score = query @ key.transpose(0, 1)  # Attention score matrix, each sample's query vector dot product with all samples' key vectors, resulting in a score matrix where each element represents attention weight from one sample to another.
            attention_scores_list.append(attention_score)

        attention_scores = torch.stack(attention_scores_list, dim=0).mean(0)  # [N,N] no longer softmax

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention, attention_scores

class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x@x.T
    mask = torch.eye(x_dis.shape[0]).to(x.device)
    x_sum = torch.sum(x**2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis*(x_sum**(-1))
    x_dis = (1-mask) * x_dis
    return x_dis

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class EncoderLayer(nn.Module):
    def __init__(self, args, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.effectattn = EfficientAttention(in_channels=d_model, key_channels=d_model, head_count=heads, value_channels=d_model)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.norm_1(x)
        x_pre, attention_scores = self.effectattn(x2)  # Get attention score matrix
        x = x + x_pre
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, attention_scores


class RNCGLN_model(nn.Module):
    def __init__(self, arg):
        super(RNCGLN_model, self).__init__()
        self.dropout = arg.random_aug_feature
        self.Trans_layer_num = arg.Trans_layer_num
        self.layers = get_clones(
            EncoderLayer(arg, arg.trans_dim, arg.nheads, arg.dropout_att),
            self.Trans_layer_num
        )
        self.norm_input = Norm(arg.ft_size)
        self.MLPfirst = nn.Linear(arg.ft_size, arg.trans_dim)
        self.norm_layer = Norm(arg.trans_dim)
        self.MLPlast = nn.Linear(arg.trans_dim, arg.nclasses)

    def forward(self, x_input):
        #x_input = self.norm_input(x_input)
        x = self.MLPfirst(x_input)
        x = F.dropout(x, self.dropout, training=self.training)
        x_dis = get_feature_dis(self.norm_layer(x))

        attention_scores = None
        for i in range(self.Trans_layer_num):
            x, layer_attention_scores = self.layers[i](x)
            if attention_scores is None:
                attention_scores = layer_attention_scores
            else:
                attention_scores += layer_attention_scores
        attention_scores /= self.Trans_layer_num  # Average multi-head attention

        CONN_INDEX = self.MLPlast(x)
        return CONN_INDEX, x_dis, attention_scores