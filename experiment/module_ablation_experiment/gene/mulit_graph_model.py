import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from model.DGSO.edge_noise_model import (RNCGLN_model, get_A_r, Ncontrast,
                                         accuracy, evaluate_metrics, monitor_score,
                                         fix_isolated_nodes_dense, robust_contrastive_loss)
from mulit_GCN import Multi_GNN_model
import copy
import pandas as pd

class Mulit_Graph_model:
    def __init__(self, gene_features,
                 gene_adj,
                 labels, idx_train, idx_val, idx_test, args):
        #节点特征
        self.gene_features = gene_features.to(args.device)

        #邻接矩阵
        self.gene_adj = gene_adj

        #标签
        self.labels = labels.to(args.device)
        #训练集，验证集，测试集id
        self.idx_train = idx_train.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)

        self.args = args

        # 初始化Multi_GNN_model网络
        self.multimodal_gcn = Multi_GNN_model(
            gene_num_features=gene_features.size(1),
            num_classes=labels.max().item() + 1,
            gene_edge_index=gene_adj,
            gene_x=gene_features,
            args=args).to(args.device)

        # 初始化RNCGLN_model
        self.model = RNCGLN_model(self.args).to(args.device)

        # 存储融合后的特征和邻接矩阵
        self.fused_features = None
        self.fused_adj = None
        self.adjust_fused_adj = None #存储动态调整的邻接矩阵

        # ---------- 早停 ----------
        self.best_state = None  # 用于保存最佳权重

        # self.best_val_metrics = None  # 保存“最佳权重”当时对应的验证集指标
        self.test_metrics_history = []  # 记录每次测试集评估的结果


    def training(self):
        print("Started training...")
        optimiser = torch.optim.Adam(list(self.multimodal_gcn.parameters()) + list(self.model.parameters()), lr=self.args.lr, weight_decay=self.args.wd)

        #如果是bool，则将它们转换为整数类型的索引张量
        if self.idx_train.dtype == torch.bool:
            self.idx_train = torch.where(self.idx_train == 1)[0]
            self.idx_val = torch.where(self.idx_val == 1)[0]
            self.idx_test = torch.where(self.idx_test == 1)[0]

        train_lbls = self.labels[self.idx_train]
        val_lbls = self.labels[self.idx_val]
        test_lbls = self.labels[self.idx_test]

        cnt_wait = 0
        best = 1e-9
        stop_epoch = 0
        start = time.time()
        totalL = []

        for epoch in range(self.args.nb_epochs):
            self.multimodal_gcn.train()
            self.model.train()
            optimiser.zero_grad()
            # 三个GCN和融合模块
            logits,gene_log,self.fused_features, self.fused_adj = self.multimodal_gcn(
                self.gene_features,
                self.gene_adj)

            #如果调整后的邻接矩阵列表不是空那么就得用调整后的，动态调整第二部分网络的邻接矩阵
            if self.adjust_fused_adj is not None:
                self.fused_adj = self.adjust_fused_adj

            # --- 统一邻接矩阵为 [N, N] 方阵 ---
            if self.fused_adj.dim() == 2 and self.fused_adj.size(0) == self.fused_adj.size(1):
                # 已经是方阵
                fused_adj_dense = self.fused_adj.float()
            else:
                # 边索引格式 [2, E]
                num_nodes = self.fused_features.size(0)
                edge_index = self.fused_adj  # [2, E]
                fused_adj_dense = torch.zeros((num_nodes, num_nodes), device=self.args.device)
                fused_adj_dense[edge_index[0], edge_index[1]] = 1.0
                fused_adj_dense[edge_index[1], edge_index[0]] = 1.0  # 无向图对称

            # 传递融合后的特征和邻接矩阵
            embeds_tra, x_dis, attention_scores = self.model(self.fused_features)
            #loss_cla = F.cross_entropy(embeds_tra[self.idx_train], self.labels_oneHot[self.idx_train])
            loss_cla = F.cross_entropy(embeds_tra[self.idx_train], self.labels[self.idx_train])

            #计算gcnmf损失 pet_log, pathology_log, gene_log, extra_log
            # 计算损失
            pet_loss = F.nll_loss(gene_log[self.idx_train], self.labels[self.idx_train])
            # pathology_loss = F.nll_loss(pathology_log[self.idx_train], self.labels[self.idx_train])
            # gene_loss = F.nll_loss(gene_log[self.idx_train], self.labels[self.idx_train])
            # extra_loss = F.nll_loss(extra_log[self.idx_train], self.labels[self.idx_train])
            #loss_gcn = pet_loss + pathology_loss + gene_loss + extra_loss

            # 计算融合损失
            logits_log_softmax = F.log_softmax(logits[self.idx_train], dim=1)
            loss_fuse = F.nll_loss(logits_log_softmax, self.labels[self.idx_train])

            # 计算损失
            loss_Ncontrast = robust_contrastive_loss(fused_adj_dense, x_dis, embeds_tra, temperature=self.args.tau)
            loss = loss_cla + self.args.r1 * loss_Ncontrast + self.args.r2 * loss_fuse #+ self.args.r3 * loss_gcn

            # 训练集评估
            if epoch % 5 == 0 and epoch != 0:
                print(f"total_loss {loss}，loss_cla {loss_cla},loss_Ncontrast {loss_Ncontrast},loss_fuse {loss_fuse}")

            loss.backward()
            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0:
                totalL.append(loss.item())
                self.multimodal_gcn.eval()
                self.model.eval()
                # 先通过 MultiModalGCN 获取融合特征
                _,_, fused_features_eval, fused_adj_eval = self.multimodal_gcn(self.gene_features,
                                                                             self.gene_adj)
                embeds_eval, x_dis_eval, attention_scores_eval = self.model(fused_features_eval)

                # 验证集评估
                val_preds = embeds_eval[self.idx_val].max(1)[1].detach().cpu().numpy()  # 获取验证集预测结果
                val_true = val_lbls.cpu().numpy()  # 获取验证集真实标签
                val_prob = embeds_eval[self.idx_val].softmax(dim=1).detach().cpu().numpy()  # 获取验证集预测概率
                val_metrics = evaluate_metrics(val_true, val_preds, val_prob)

                # 训练集评估
                tra_preds = embeds_eval[self.idx_train].max(1)[1].detach().cpu().numpy()  # 获取训练集预测结果
                tra_true = train_lbls.cpu().numpy()  # 获取训练集真实标签
                tra_prob = embeds_eval[self.idx_train].softmax(dim=1).detach().cpu().numpy()  # 获取训练集预测概率
                tra_metrics = evaluate_metrics(tra_true, tra_preds, tra_prob)

                # 保存测试集结果，调试用
                test_preds = embeds_eval[self.idx_test].max(1)[1].detach().cpu().numpy()  # 获取测试集预测结果
                test_true = test_lbls.cpu().numpy()  # 获取测试集真实标签
                test_prob = embeds_eval[self.idx_test].softmax(dim=1).detach().cpu().numpy()  # 获取测试集预测概率
                test_metrics = evaluate_metrics(test_true, test_preds, test_prob)
                self.test_metrics_history.append(test_metrics)

                # 打印训练集和验证集的评估指标
                print(f"Epoch {epoch}:")
                print(
                    f"Train - ACC: {tra_metrics['ACC']:.4f}, AUC: {tra_metrics['AUC']:.4f}, F1: {tra_metrics['F1']:.4f}, RECALL: {tra_metrics['RECALL']:.4f}")
                print(
                    f"Validation - ACC: {val_metrics['ACC']:.4f}, AUC: {val_metrics['AUC']:.4f}, F1: {val_metrics['F1']:.4f}, RECALL: {val_metrics['RECALL']:.4f}")
                print(f"Validation Confusion Matrix:\n{val_metrics['Confusion Matrix']}")

                stop_epoch = epoch
                score = monitor_score(val_metrics)
                #if score >= best and epoch >= self.args.warmup_num:
                if score >= best:
                    best = score
                    cnt_wait = 0
                    self.best_state = {
                        'multimodal_gcn': copy.deepcopy(self.multimodal_gcn.state_dict()),
                        'model': copy.deepcopy(self.model.state_dict())
                    }
                    if epoch >= self.args.warmup_num:
                        # ---------- 1. 计算全图的可靠性分数 ----------
                        alpha = 0.5
                        torch.diagonal(x_dis_eval).fill_(1.0)  # 将对角线元素设置为1
                        # 首先将 x_dis_eval 和 attention_scores_eval 分别归一化到 [0, 1] 范围内
                        x_dis_eval_norm = (x_dis_eval - x_dis_eval.min()) / (x_dis_eval.max() - x_dis_eval.min() + 1e-8)
                        attention_scores_eval_norm = (attention_scores_eval - attention_scores_eval.min()) / (
                                    attention_scores_eval.max() - attention_scores_eval.min() + 1e-8)

                        # 计算可靠性分数
                        reliability_score = x_dis_eval_norm + alpha * attention_scores_eval_norm

                        # 再次归一化 reliability_score 到 [0, 1] 范围内
                        reliability_score = (reliability_score - reliability_score.min()) / (
                                    reliability_score.max() - reliability_score.min() + 1e-8)

                        # 计算每个节点的噪声密度
                        # 每个节点的边的可靠性分数的平均值
                        noise_density = reliability_score.sum(dim=1) / reliability_score.size(1)  # 计算每个节点所有边的可靠性分数的平均值

                        # 将 noise_density 归一化到 [0, 1] 范围内
                        noise_density = (noise_density - noise_density.min()) / (
                                    noise_density.max() - noise_density.min() + 1e-8)
                        noise_density = noise_density.clamp(min=0, max=1)  # 限制在 [0, 1] 范围内

                        # 设置相似度阈值
                        sim_threshold_2 = 0.8  # 当其中一个节点噪声密度高于阈值时使用的相似度阈值
                        noise_density_threshold = 0.8 # 噪声密度的阈值

                        # 初始化新的邻接矩阵
                        pseudo_adj_label = torch.zeros_like(reliability_score)

                        # 遍历每个节点
                        for i in range(noise_density.size(0)):
                            # 获取当前节点与其他所有节点的相似度
                            similarities = x_dis_eval[i].clone()  # 克隆一份，避免修改原数据

                            # 排除自身相似度（自身相似度通常最高，需要排除）
                            similarities[i] = -1  # 设置为一个极小值，确保不会被选中

                            # 获取相似度最高的5个节点索引（处理并列情况）
                            _, top_indices = torch.topk(similarities, k=5, largest=True)

                            # 如果存在并列情况，找到所有并列的节点
                            top_similarities = similarities[top_indices]
                            threshold_similarity = top_similarities.min()  # 最小的前5个相似度值
                            all_top_indices = torch.where(similarities >= threshold_similarity)[0]

                            # 遍历这些最相似的节点
                            for j in all_top_indices:
                                # 如果当前节点是噪声节点，且相似度小于阈值，则跳过
                                if noise_density[i] > noise_density_threshold and x_dis_eval_norm[i, j] < sim_threshold_2:
                                    continue
                                # 如果目标节点是噪声节点，且相似度小于阈值，则跳过
                                if noise_density[j] > noise_density_threshold and x_dis_eval_norm[i, j] < sim_threshold_2:
                                    continue
                                # 建立边
                                pseudo_adj_label[i, j] = 1
                                pseudo_adj_label[j, i] = 1  # 无向图，需要添加反向边

                        # 更新邻接矩阵为伪图的邻接矩阵
                        self.adjust_fused_adj = pseudo_adj_label  # 更新融合后的邻接矩阵
                        # 修复孤立节点
                        self.adjust_fused_adj = fix_isolated_nodes_dense(pseudo_adj_label, x_dis_eval)
                        print("\n --> A new loop after pseudo-graph update")
                else:
                    if epoch > self.args.warmup_num:
                        cnt_wait += 5
                if cnt_wait >= self.args.patience:
                    if self.best_state is not None:
                        self.multimodal_gcn.load_state_dict(self.best_state['multimodal_gcn'])
                        self.model.load_state_dict(self.best_state['model'])
                    break
            ################STA|Eval|###############


        training_time = time.time() - start

        # === 最终测试集评估 ===
        self.multimodal_gcn.eval()
        self.model.eval()

        with torch.no_grad():
            _,_, fused_features_final, fused_adj_final = self.multimodal_gcn(
                self.gene_features,
                self.gene_adj)

            embeds_final, _, _ = self.model(fused_features_final)

            test_preds = embeds_final[self.idx_test].max(1)[1].detach().cpu().numpy()
            test_true = test_lbls.cpu().numpy()
            test_prob = embeds_final[self.idx_test].softmax(dim=1).detach().cpu().numpy()
            test_metrics = evaluate_metrics(test_true, test_preds, test_prob)
            # 获取测试集节点编号
            test_node_ids = self.idx_test.cpu().numpy()

            # 创建一个 DataFrame 来存储节点编号、预测概率和真实标签
            df_results = pd.DataFrame({
                'Node_ID': test_node_ids,
                'True_Label': test_true,
                'Predicted_Class': test_preds
            })

            # 添加预测概率列
            for i in range(test_prob.shape[1]):
                df_results[f'Prob_Class_{i}'] = test_prob[:, i]
            # 打印结果
            print("\n === 测试集结果 ===")
            print(df_results)

        print("\n stop_epoch: {:}| training_time: {:.4f} \n".format(stop_epoch, training_time))

        return training_time, stop_epoch,test_metrics
