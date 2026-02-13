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
    def __init__(self, pet_features, pathology_features, gene_features, extra_features,
                 pet_adj, pathology_adj,gene_adj,extra_adj,
                 labels, idx_train, idx_val, idx_test, args):

        self.pet_features = pet_features.to(args.device)
        self.pathology_features = pathology_features.to(args.device)
        self.gene_features = gene_features.to(args.device)
        self.extra_features = extra_features.to(args.device)

        self.pet_adj = pet_adj
        self.pathology_adj = pathology_adj
        self.gene_adj = gene_adj
        self.extra_adj = extra_adj

        self.labels = labels.to(args.device)

        self.idx_train = idx_train.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)

        self.args = args


        self.multimodal_gcn = Multi_GNN_model(
            pet_num_features=pet_features.size(1),pathology_num_features=pathology_features.size(1),gene_num_features=gene_features.size(1),extra_num_features=extra_features.size(1),
            num_classes=labels.max().item() + 1,
            pet_edge_index=pet_adj,pathology_edge_index=pathology_adj,gene_edge_index=gene_adj,extra_edge_index=extra_adj,
            pet_x=pet_features,pathology_x=pathology_features,gene_x=gene_features,extra_x=extra_features,
            args=args).to(args.device)


        self.model = RNCGLN_model(self.args).to(args.device)


        self.fused_features = None
        self.fused_adj = None
        self.adjust_fused_adj = None


        self.best_state = None


        self.test_metrics_history = []


    def training(self):
        print("Started training...")
        optimiser = torch.optim.Adam(list(self.multimodal_gcn.parameters()) + list(self.model.parameters()), lr=self.args.lr, weight_decay=self.args.wd)

        #
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

            logits,pet_log, pathology_log, gene_log, extra_log,self.fused_features, self.fused_adj = self.multimodal_gcn(
                self.pet_features, self.pathology_features, self.gene_features, self.extra_features,
                self.pet_adj, self.pathology_adj, self.gene_adj, self.extra_adj)


            if self.adjust_fused_adj is not None:
                self.fused_adj = self.adjust_fused_adj


            if self.fused_adj.dim() == 2 and self.fused_adj.size(0) == self.fused_adj.size(1):

                fused_adj_dense = self.fused_adj.float()
            else:

                num_nodes = self.fused_features.size(0)
                edge_index = self.fused_adj  # [2, E]
                fused_adj_dense = torch.zeros((num_nodes, num_nodes), device=self.args.device)
                fused_adj_dense[edge_index[0], edge_index[1]] = 1.0
                fused_adj_dense[edge_index[1], edge_index[0]] = 1.0


            embeds_tra, x_dis, attention_scores = self.model(self.fused_features)
            loss_cla = F.cross_entropy(embeds_tra[self.idx_train], self.labels[self.idx_train])

            pet_loss = F.nll_loss(pet_log[self.idx_train], self.labels[self.idx_train])
            pathology_loss = F.nll_loss(pathology_log[self.idx_train], self.labels[self.idx_train])
            gene_loss = F.nll_loss(gene_log[self.idx_train], self.labels[self.idx_train])
            extra_loss = F.nll_loss(extra_log[self.idx_train], self.labels[self.idx_train])
            loss_gcn = pet_loss + pathology_loss + gene_loss + extra_loss

            logits_log_softmax = F.log_softmax(logits[self.idx_train], dim=1)
            loss_fuse = F.nll_loss(logits_log_softmax, self.labels[self.idx_train])

            loss_Ncontrast = robust_contrastive_loss(fused_adj_dense, x_dis, embeds_tra, temperature=self.args.tau)
            loss = loss_cla + self.args.r1 * loss_Ncontrast + self.args.r2 * loss_fuse + self.args.r3 * loss_gcn

            if epoch % 5 == 0 and epoch != 0:
                print(f"total_loss {loss}，loss_cla {loss_cla},loss_Ncontrast {loss_Ncontrast},loss_fuse {loss_fuse},loss_gcn {loss_gcn}")

            loss.backward()
            optimiser.step()

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0:
                totalL.append(loss.item())
                self.multimodal_gcn.eval()
                self.model.eval()
                _,_,_,_,_, fused_features_eval, fused_adj_eval = self.multimodal_gcn(self.pet_features, self.pathology_features,
                                                                             self.gene_features,self.extra_features,
                                                                             self.pet_adj, self.pathology_adj,
                                                                             self.gene_adj, self.extra_adj)
                embeds_eval, x_dis_eval, attention_scores_eval = self.model(fused_features_eval)

                val_preds = embeds_eval[self.idx_val].max(1)[1].detach().cpu().numpy()
                val_true = val_lbls.cpu().numpy()
                val_prob = embeds_eval[self.idx_val].softmax(dim=1).detach().cpu().numpy()
                val_metrics = evaluate_metrics(val_true, val_preds, val_prob)

                tra_preds = embeds_eval[self.idx_train].max(1)[1].detach().cpu().numpy()
                tra_true = train_lbls.cpu().numpy()
                tra_prob = embeds_eval[self.idx_train].softmax(dim=1).detach().cpu().numpy()
                tra_metrics = evaluate_metrics(tra_true, tra_preds, tra_prob)

                test_preds = embeds_eval[self.idx_test].max(1)[1].detach().cpu().numpy()
                test_true = test_lbls.cpu().numpy()
                test_prob = embeds_eval[self.idx_test].softmax(dim=1).detach().cpu().numpy()
                test_metrics = evaluate_metrics(test_true, test_preds, test_prob)
                self.test_metrics_history.append(test_metrics)

                print(f"Epoch {epoch}:")
                print(
                    f"Train - ACC: {tra_metrics['ACC']:.4f}, AUC: {tra_metrics['AUC']:.4f}, F1: {tra_metrics['F1']:.4f}, RECALL: {tra_metrics['RECALL']:.4f}")
                print(
                    f"Validation - ACC: {val_metrics['ACC']:.4f}, AUC: {val_metrics['AUC']:.4f}, F1: {val_metrics['F1']:.4f}, RECALL: {val_metrics['RECALL']:.4f}")
                print(f"Validation Confusion Matrix:\n{val_metrics['Confusion Matrix']}")

                stop_epoch = epoch
                score = monitor_score(val_metrics)
                if score >= best:
                    best = score
                    cnt_wait = 0
                    self.best_state = {
                        'multimodal_gcn': copy.deepcopy(self.multimodal_gcn.state_dict()),
                        'model': copy.deepcopy(self.model.state_dict())
                    }
                    if epoch >= self.args.warmup_num:
                        alpha = 0.5
                        torch.diagonal(x_dis_eval).fill_(1.0)
                        x_dis_eval_norm = (x_dis_eval - x_dis_eval.min()) / (x_dis_eval.max() - x_dis_eval.min() + 1e-8)
                        attention_scores_eval_norm = (attention_scores_eval - attention_scores_eval.min()) / (
                                    attention_scores_eval.max() - attention_scores_eval.min() + 1e-8)

                        reliability_score = x_dis_eval_norm + alpha * attention_scores_eval_norm

                        reliability_score = (reliability_score - reliability_score.min()) / (
                                    reliability_score.max() - reliability_score.min() + 1e-8)

                        noise_density = reliability_score.sum(dim=1) / reliability_score.size(1)

                        noise_density = (noise_density - noise_density.min()) / (
                                    noise_density.max() - noise_density.min() + 1e-8)
                        noise_density = noise_density.clamp(min=0, max=1)

                        sim_threshold_2 = 0.8
                        noise_density_threshold = 0.8

                        pseudo_adj_label = torch.zeros_like(reliability_score)

                        for i in range(noise_density.size(0)):
                            similarities = x_dis_eval[i].clone()

                            similarities[i] = -1

                            _, top_indices = torch.topk(similarities, k=5, largest=True)

                            top_similarities = similarities[top_indices]
                            threshold_similarity = top_similarities.min()
                            all_top_indices = torch.where(similarities >= threshold_similarity)[0]

                            for j in all_top_indices:
                                if noise_density[i] > noise_density_threshold and x_dis_eval_norm[i, j] < sim_threshold_2:
                                    continue
                                if noise_density[j] > noise_density_threshold and x_dis_eval_norm[i, j] < sim_threshold_2:
                                    continue
                                pseudo_adj_label[i, j] = 1
                                pseudo_adj_label[j, i] = 1

                        self.adjust_fused_adj = pseudo_adj_label
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

        self.multimodal_gcn.eval()
        self.model.eval()

        with torch.no_grad():
            _,_,_,_,_, fused_features_final, fused_adj_final = self.multimodal_gcn(
                self.pet_features, self.pathology_features, self.gene_features,self.extra_features,
                self.pet_adj, self.pathology_adj, self.gene_adj,self.extra_adj)

            embeds_final, _, _ = self.model(fused_features_final)

            test_preds = embeds_final[self.idx_test].max(1)[1].detach().cpu().numpy()
            test_true = test_lbls.cpu().numpy()
            test_prob = embeds_final[self.idx_test].softmax(dim=1).detach().cpu().numpy()
            test_metrics = evaluate_metrics(test_true, test_preds, test_prob)
            test_node_ids = self.idx_test.cpu().numpy()

            df_results = pd.DataFrame({
                'Node_ID': test_node_ids,
                'True_Label': test_true,
                'Predicted_Class': test_preds
            })

            for i in range(test_prob.shape[1]):
                df_results[f'Prob_Class_{i}'] = test_prob[:, i]
            print("\n === 测试集结果 ===")
            print(df_results)

        print("\n stop_epoch: {:}| training_time: {:.4f} \n".format(stop_epoch, training_time))

        return training_time, stop_epoch,test_metrics
