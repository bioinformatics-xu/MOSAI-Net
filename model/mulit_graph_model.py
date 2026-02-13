import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from model.DGSO.edge_noise_model import (RNCGLN_model, get_A_r, Ncontrast,
                                         accuracy, evaluate_metrics, monitor_score,
                                         fix_isolated_nodes_dense, robust_contrastive_loss)
from model.AMSF.mulit_GCN import Multi_GNN_model
import copy
import pandas as pd
import os

class Mulit_Graph_model:
    def __init__(self, pet_features, pathology_features, gene_features, extra_features,
                 pet_adj, pathology_adj,gene_adj,extra_adj,
                 labels, idx_train, idx_val, idx_test, args, fold_id):

        # Add fold_id parameter
        self.fold_id = fold_id
        self.best_epoch = None

        self.modality_ratio_history = []  # Record modality ratio per epoch

        # Node features
        self.pet_features = pet_features.to(args.device)
        self.pathology_features = pathology_features.to(args.device)
        self.gene_features = gene_features.to(args.device)
        self.extra_features = extra_features.to(args.device)
        # Adjacency matrices
        self.pet_adj = pet_adj
        self.pathology_adj = pathology_adj
        self.gene_adj = gene_adj
        self.extra_adj = extra_adj
        # Labels
        self.labels = labels.to(args.device)
        # Training, validation, test set indices
        self.idx_train = idx_train.to(args.device)
        self.idx_val = idx_val.to(args.device)
        self.idx_test = idx_test.to(args.device)

        self.args = args

        # Initialize Multi_GNN_model network
        self.multimodal_gcn = Multi_GNN_model(
            pet_num_features=pet_features.size(1),pathology_num_features=pathology_features.size(1),gene_num_features=gene_features.size(1),extra_num_features=extra_features.size(1),
            num_classes=labels.max().item() + 1,
            pet_edge_index=pet_adj,pathology_edge_index=pathology_adj,gene_edge_index=gene_adj,extra_edge_index=extra_adj,
            pet_x=pet_features,pathology_x=pathology_features,gene_x=gene_features,extra_x=extra_features,
            args=args).to(args.device)

        # Initialize RNCGLN_model
        self.model = RNCGLN_model(self.args).to(args.device)

        # Store fused features and adjacency matrix
        self.fused_features = None
        self.fused_adj = None
        self.adjust_fused_adj = None  # Store dynamically adjusted adjacency matrix

        # ---------- Early stopping ----------
        self.best_state = None  # For saving best weights

        # self.best_val_metrics = None  # Save validation metrics corresponding to best weights
        self.test_metrics_history = []  # Record test set evaluation results

    def training(self):
        print("Started training...")
        optimiser = torch.optim.Adam(list(self.multimodal_gcn.parameters()) + list(self.model.parameters()), lr=self.args.lr, weight_decay=self.args.wd)

        # Convert to integer index tensors if they are boolean
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
            # Three GCNs and fusion module
            logits,pet_log, pathology_log, gene_log, extra_log,self.fused_features, self.fused_adj = self.multimodal_gcn(
                self.pet_features, self.pathology_features, self.gene_features, self.extra_features,
                self.pet_adj, self.pathology_adj, self.gene_adj, self.extra_adj)

            # If adjusted adjacency matrix list is not empty, use adjusted one for dynamic adjustment of second network's adjacency matrix
            if self.adjust_fused_adj is not None:
                self.fused_adj = self.adjust_fused_adj

            # --- Unify adjacency matrix to [N, N] square matrix ---
            if self.fused_adj.dim() == 2 and self.fused_adj.size(0) == self.fused_adj.size(1):
                # Already square matrix
                fused_adj_dense = self.fused_adj.float()
            else:
                # Edge index format [2, E]
                num_nodes = self.fused_features.size(0)
                edge_index = self.fused_adj  # [2, E]
                fused_adj_dense = torch.zeros((num_nodes, num_nodes), device=self.args.device)
                fused_adj_dense[edge_index[0], edge_index[1]] = 1.0
                fused_adj_dense[edge_index[1], edge_index[0]] = 1.0  # Undirected graph symmetry

            # Pass fused features and adjacency matrix
            embeds_tra, x_dis, attention_scores = self.model(self.fused_features)
            #loss_cla = F.cross_entropy(embeds_tra[self.idx_train], self.labels_oneHot[self.idx_train])
            loss_cla = F.cross_entropy(embeds_tra[self.idx_train], self.labels[self.idx_train])

            # Calculate gcnmf loss pet_log, pathology_log, gene_log, extra_log
            # Calculate loss
            pet_loss = F.nll_loss(pet_log[self.idx_train], self.labels[self.idx_train])
            pathology_loss = F.nll_loss(pathology_log[self.idx_train], self.labels[self.idx_train])
            gene_loss = F.nll_loss(gene_log[self.idx_train], self.labels[self.idx_train])
            extra_loss = F.nll_loss(extra_log[self.idx_train], self.labels[self.idx_train])
            loss_gcn = pet_loss + pathology_loss + gene_loss + extra_loss

            # Calculate fusion loss
            logits_log_softmax = F.log_softmax(logits[self.idx_train], dim=1)
            loss_fuse = F.nll_loss(logits_log_softmax, self.labels[self.idx_train])

            # Calculate loss
            loss_Ncontrast = robust_contrastive_loss(fused_adj_dense, x_dis, embeds_tra, temperature=self.args.tau)
            loss = loss_cla + self.args.r1 * loss_Ncontrast + self.args.r2 * loss_fuse + self.args.r3 * loss_gcn

            # Training set evaluation
            if epoch % 5 == 0 and epoch != 0:
                print(f"total_loss {loss}，loss_cla {loss_cla},loss_Ncontrast {loss_Ncontrast},loss_fuse {loss_fuse},loss_gcn {loss_gcn}")

            loss.backward()
            optimiser.step()

            if epoch == 10:
                # Save original adjacency matrix
                base_path = r"experiment/CV-adjacency_matrix"
                fold_folder = os.path.join(base_path, f"fold_{self.fold_id}")
                os.makedirs(fold_folder, exist_ok=True)  # Create folder if it doesn't exist

                # Save original fused adjacency matrix
                original_fused_adj = self.fused_adj.clone()  # Clone original fused adjacency matrix
                adj_file_name = os.path.join(fold_folder, f"original_adj_matrix.pth")  # Filename
                torch.save(original_fused_adj, adj_file_name)  # Save original adjacency matrix
                print(f"Original adjacency matrix saved to: {adj_file_name}")

            ################STA|Eval|###############
            if epoch % 5 == 0 and epoch != 0:
                totalL.append(loss.item())
                self.multimodal_gcn.eval()
                self.model.eval()
                # First get fused features through MultiModalGCN
                _,_,_,_,_, fused_features_eval, fused_adj_eval = self.multimodal_gcn(self.pet_features, self.pathology_features,
                                                                             self.gene_features,self.extra_features,
                                                                             self.pet_adj, self.pathology_adj,
                                                                             self.gene_adj, self.extra_adj)
                embeds_eval, x_dis_eval, attention_scores_eval = self.model(fused_features_eval)

                # Validation set evaluation
                val_preds = embeds_eval[self.idx_val].max(1)[1].detach().cpu().numpy()  # Get validation predictions
                val_true = val_lbls.cpu().numpy()  # Get validation true labels
                val_prob = embeds_eval[self.idx_val].softmax(dim=1).detach().cpu().numpy()  # Get validation prediction probabilities
                val_metrics = evaluate_metrics(val_true, val_preds, val_prob)

                # Training set evaluation
                tra_preds = embeds_eval[self.idx_train].max(1)[1].detach().cpu().numpy()  # Get training predictions
                tra_true = train_lbls.cpu().numpy()  # Get training true labels
                tra_prob = embeds_eval[self.idx_train].softmax(dim=1).detach().cpu().numpy()  # Get training prediction probabilities
                tra_metrics = evaluate_metrics(tra_true, tra_preds, tra_prob)

                # Save test set results for debugging
                test_preds = embeds_eval[self.idx_test].max(1)[1].detach().cpu().numpy()  # Get test predictions
                test_true = test_lbls.cpu().numpy()  # Get test true labels
                test_prob = embeds_eval[self.idx_test].softmax(dim=1).detach().cpu().numpy()  # Get test prediction probabilities
                test_metrics = evaluate_metrics(test_true, test_preds, test_prob)
                self.test_metrics_history.append(test_metrics)

                # Print training and validation metrics
                print(f"Epoch {epoch}:")
                print(
                    f"Train - ACC: {tra_metrics['ACC']:.4f}, AUC: {tra_metrics['AUC']:.4f}, F1: {tra_metrics['F1']:.4f}, RECALL: {tra_metrics['RECALL']:.4f}")
                print(
                    f"Validation - ACC: {val_metrics['ACC']:.4f}, AUC: {val_metrics['AUC']:.4f}, F1: {val_metrics['F1']:.4f}, RECALL: {val_metrics['RECALL']:.4f}")
                print(f"Validation Confusion Matrix:\n{val_metrics['Confusion Matrix']}")

                ################ Record global modality ratio ################
                self.multimodal_gcn.eval()
                with torch.no_grad():
                    # Re-get 4 GCN outputs (variables already exist, no need to recalculate)
                    _, pet_x_gcn = self.multimodal_gcn.gcn_pet(self.pet_features, self.pet_adj)
                    _, pathology_x_gcn = self.multimodal_gcn.gcn_pathology(self.pathology_features, self.pathology_adj)
                    _, gene_x_gcn = self.multimodal_gcn.gcn_gene(self.gene_features, self.gene_adj)
                    _, extra_x_gcn = self.multimodal_gcn.gcn_extra(self.extra_features, self.extra_adj)

                    # Only get validation set nodes
                    pet_x_gcn_val = pet_x_gcn[self.idx_val]
                    patho_x_gcn_val = pathology_x_gcn[self.idx_val]
                    gene_x_gcn_val = gene_x_gcn[self.idx_val]
                    extra_x_gcn_val = extra_x_gcn[self.idx_val]

                    ratios = self.multimodal_gcn.get_modality_usage_ratio(
                        pet_x_gcn_val, patho_x_gcn_val, gene_x_gcn_val, extra_x_gcn_val
                    )

                    self.modality_ratio_history.append({
                        'epoch': epoch,
                        'PET_ratio': ratios[0],
                        'Pathology_ratio': ratios[1],
                        'Gene_ratio': ratios[2],
                        'Extra_ratio': ratios[3]
                    })

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
                    # Record epoch corresponding to best weights
                    self.best_epoch = epoch  # Add attribute to record epoch of best weights
                    print(f"Updated best weights, current epoch: {epoch}")

                    if epoch >= self.args.warmup_num:
                        # ---------- 1. Calculate reliability scores for entire graph ----------
                        alpha = 0.5
                        torch.diagonal(x_dis_eval).fill_(1.0)  # Set diagonal elements to 1
                        # First normalize x_dis_eval and attention_scores_eval to [0, 1] range
                        x_dis_eval_norm = (x_dis_eval - x_dis_eval.min()) / (x_dis_eval.max() - x_dis_eval.min() + 1e-8)
                        attention_scores_eval_norm = (attention_scores_eval - attention_scores_eval.min()) / (
                                    attention_scores_eval.max() - attention_scores_eval.min() + 1e-8)

                        # Calculate reliability score
                        reliability_score = x_dis_eval_norm + alpha * attention_scores_eval_norm

                        # Normalize reliability_score to [0, 1] range again
                        reliability_score = (reliability_score - reliability_score.min()) / (
                                    reliability_score.max() - reliability_score.min() + 1e-8)

                        # Calculate noise density for each node
                        # Average reliability score of edges for each node
                        noise_density = reliability_score.sum(dim=1) / reliability_score.size(1)  # Calculate average reliability score of all edges for each node

                        # Normalize noise_density to [0, 1] range
                        noise_density = (noise_density - noise_density.min()) / (
                                    noise_density.max() - noise_density.min() + 1e-8)
                        noise_density = noise_density.clamp(min=0, max=1)  # Limit to [0, 1] range

                        # Set similarity threshold
                        sim_threshold_2 = 0.8  # Similarity threshold used when one node's noise density is above threshold
                        noise_density_threshold = 0.8  # Noise density threshold

                        # Initialize new adjacency matrix
                        pseudo_adj_label = torch.zeros_like(reliability_score)

                        # Iterate through each node
                        for i in range(noise_density.size(0)):
                            # Get similarity between current node and all other nodes
                            similarities = x_dis_eval[i].clone()  # Clone to avoid modifying original data

                            # Exclude self-similarity (self-similarity is usually highest, needs to be excluded)
                            similarities[i] = -1  # Set to minimum value to ensure not selected

                            # Get indices of top 5 most similar nodes (handle ties)
                            _, top_indices = torch.topk(similarities, k=5, largest=True)

                            # If ties exist, find all tied nodes
                            top_similarities = similarities[top_indices]
                            threshold_similarity = top_similarities.min()  # Minimum top 5 similarity value
                            all_top_indices = torch.where(similarities >= threshold_similarity)[0]

                            # Iterate through these most similar nodes
                            for j in all_top_indices:
                                # If current node is noise node and similarity is below threshold, skip
                                if noise_density[i] > noise_density_threshold and x_dis_eval_norm[i, j] < sim_threshold_2:
                                    continue
                                # If target node is noise node and similarity is below threshold, skip
                                if noise_density[j] > noise_density_threshold and x_dis_eval_norm[i, j] < sim_threshold_2:
                                    continue
                                # Establish edge
                                pseudo_adj_label[i, j] = 1
                                pseudo_adj_label[j, i] = 1  # Undirected graph, need to add reverse edge

                        # Update adjacency matrix to pseudo-graph adjacency matrix
                        self.adjust_fused_adj = pseudo_adj_label
                        # Fix isolated nodes
                        self.adjust_fused_adj = fix_isolated_nodes_dense(pseudo_adj_label, x_dis_eval)

                        # Specify a base path (absolute path)
                        base_path = r"experiment/CV-adjacency_matrix"
                        # Create folder based on fold_id
                        fold_folder = os.path.join(base_path, f"fold_{self.fold_id}")
                        os.makedirs(fold_folder, exist_ok=True)  # Create folder if it doesn't exist
                        # Save updated adjacency matrix to file
                        adj_file_name = os.path.join(fold_folder, f"adj_matrix_epoch_{epoch}.pth")  # Filename includes current epoch and fold
                        torch.save(self.adjust_fused_adj, adj_file_name)  # Save adjacency matrix
                        print(f"Adjacency matrix saved to: {adj_file_name}")

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

        # === Final test set evaluation ===
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
            # Get test set node indices
            test_node_ids = self.idx_test.cpu().numpy()

            # Create DataFrame to store node indices, prediction probabilities and true labels
            df_results = pd.DataFrame({
                'Node_ID': test_node_ids,
                'True_Label': test_true,
                'Predicted_Class': test_preds
            })

            # Add prediction probability columns
            for i in range(test_prob.shape[1]):
                df_results[f'Prob_Class_{i}'] = test_prob[:, i]
            # Print results
            print("\n === Test Set Results ===")
            print(df_results)

            # Output epoch corresponding to best weights
            if hasattr(self, 'best_epoch'):
                print(f"Epoch of best weights: {self.best_epoch}")

        print("\n stop_epoch: {:}| training_time: {:.4f} \n".format(stop_epoch, training_time))

        out_dir = r"experiment/modality_ratios"
        os.makedirs(out_dir, exist_ok=True)
        df = pd.DataFrame(self.modality_ratio_history)
        df.to_excel(os.path.join(out_dir, f"fold_{self.fold_id}_modality_ratios.xlsx"),
                    index=False)
        print(f"Modality ratios saved to fold_{self.fold_id}_modality_ratios.xlsx")

        return training_time, stop_epoch,test_metrics