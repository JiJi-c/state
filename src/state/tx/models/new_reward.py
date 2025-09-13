import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection
import scanpy as sc
import anndata as ad
import pandas as pd
from hpdex import parallel_differential_expression


@torch.no_grad()
class RewardEvaluator:
    def __init__(self, gene_list: List[str],
                 pert_list: List[str],
                 reward_weights: Dict[str, float],
                 min_perts_for_pds: int = 3,
                 k_percentage: float = 0.05):
        self.gene_list = gene_list
        self.pert_list = pert_list
        self.reward_weights = reward_weights
        self.min_perts_for_pds = min_perts_for_pds
        self.k_percentage = k_percentage
        self.de_results = pd.read_csv('/yuchang/shangyue/wd/data_state/competition_support_set/de_results.csv')
        self.gene_to_idx = {gene: i for i, gene in enumerate(self.gene_list)}
        self.pert_to_idx = {pert: i for i, pert in enumerate(self.pert_list)}
        self.idx_to_pert = {i: pert for i, pert in enumerate(self.pert_list)}

    def aggregate_rewards(
        self,
        y_samples: torch.Tensor,  # K,B,L,D
        y_real: torch.Tensor,     # B,L,D
        basal_cells: torch.Tensor, # B,L,D
        pert_names: List[str],    # B*L
        mode: str = "train",
        merge: bool = False,
    ) -> torch.Tensor:
        """
        计算并组合奖励，返回每个细胞样本的最终奖励值。
        """
        K, B, L, D = y_samples.shape

        # 准备索引信息
        pert_indices = torch.tensor(
            [self.pert_to_idx.get(name, -1) for name in pert_names], 
            device=y_samples.device,
            dtype=torch.long,
        )

        local_perts_global, local_indices = pert_indices.unique(sorted=True, return_inverse=True)
        M = int(local_perts_global.numel())
        pert_indices = local_indices  # [B*L]
        
        #在validation mode下，计算每个扰动的DES分数
        if mode == "val":
            if not merge:
                mean_des, _ = self.calculate_des_reward_hpdex(
                    y_pred=y_samples.reshape(K, -1, D),                    # [K, B*L, D]
                    basal_cells=basal_cells,             # [B,L, D]
                    pert_names=pert_names,  # [B*L] 
                )
            else:
                agg_pred = self._vectorized_group_mean(
                    y_samples, pert_indices, M
                )  # [K, M, D]
                
                agg_real = self._vectorized_group_mean(
                    y_real.unsqueeze(0), pert_indices, M 
                ).squeeze(0)  # [M, D]
        
                aggregated_pert_names = [self.idx_to_pert[idx.item()] for idx in local_perts_global]
                
                mean_des, _ = self.calculate_des_reward_hpdex(
                    y_pred=agg_pred,                    # [K, M, D]
                    basal_cells=basal_cells,             # [B, L, D]
                    pert_names=aggregated_pert_names,  # [M] 
                )
            
            
        # 计算Delta值
        delta_real = y_real - basal_cells           # [B, L, D]
        delta_pred = y_samples - basal_cells.unsqueeze(0)  # [K, B, L, D]
        
        # 使用分组均值计算
        delta_real_pseudobulks = self._vectorized_group_mean(
            delta_real.unsqueeze(0), pert_indices, M
        ).squeeze(0)  # [1, M, D] → [M, D]
        
        delta_pred_pseudobulks = self._vectorized_group_mean(
            delta_pred, pert_indices, M
        )  # [K, M, D]

        mae,mae_r = self.calculate_mae_reward(delta_pred_pseudobulks, delta_real_pseudobulks) # [K, M]

        pert_names_in_batch = [self.idx_to_pert[idx.item()] for idx in local_perts_global]
        target_positions = [(i, self.gene_to_idx[name]) for i, name in enumerate(pert_names_in_batch)]
        pert_idx, gene_idx = torch.tensor(target_positions, device=y_samples.device).T  

        # 置零扰动基因
        delta_real_pseudobulks[pert_idx, gene_idx] = 0  # [M, D]
        delta_pred_pseudobulks[:, pert_idx, gene_idx] = 0  # [K, M, D]

        # 计算PDS奖励
        pds_r = self.calculate_pds_reward(delta_pred_pseudobulks, delta_real_pseudobulks) # [K, M]

        pds_w = self.reward_weights.get("pds", 1.0)
        # des_w = self.reward_weights.get("des", 1.0)
        mae_w = self.reward_weights.get("mae", 1.0)

        total_weight = pds_w + mae_w 

        rewards = (pds_w * pds_r + mae_w * mae_r) / total_weight  # [K, M] 
  
        cell_rewards = rewards[:, pert_indices].view(K, B, L)  # [K, B*L] → [K, B, L]

        mu = cell_rewards.mean(dim = 1,keepdim=True) # [K,1,L]
        std = cell_rewards.std(dim = 1,keepdim=True) + 1e-8
        cell_rewards = (cell_rewards - mu) / std # [K,B,L]

        if mode == "val":
            return cell_rewards, pds_r, mae_r, M , mae , mean_des
        return cell_rewards, pds_r, mae_r, M , mae

    def _vectorized_group_mean(
        self, values: torch.Tensor, indices: torch.Tensor, num_groups: int
    ) -> torch.Tensor:
        """
        values: [K, B, L, D]
        indices: [B*L]
        num_groups: int
        return: [K, M, D]
        """
        K, B, L, D = values.shape
    
        indices_scatter = indices.unsqueeze(0).expand(D*K, -1)  # [D*K, B*L]
        
        # Scatter求和
        sum_tensor = torch.zeros(D*K, num_groups, device=values.device, dtype=values.dtype)
        count_tensor = torch.zeros(D*K, num_groups, device=values.device, dtype=torch.long)
        
        values_scatter = values.view(K, B*L, D).permute(2, 0, 1).contiguous().view(D*K, B*L)  # [D, K, B*L]

        sum_tensor.scatter_add_(1, indices_scatter, values_scatter)
        count_tensor.scatter_add_(1, indices_scatter, torch.ones_like(values_scatter, dtype=torch.long))
        
        # 计算均值
        mean_tensor = sum_tensor / count_tensor.clamp(min=1).float()
        
        # 重塑回目标形状
        result = mean_tensor.view(D, K, num_groups).permute(1, 2, 0)  # [K, M, D]
        
        return result


    def calculate_mae_reward(
        self, pred_pseudobulks: torch.Tensor, real_pseudobulks: torch.Tensor
    ) -> torch.Tensor:
        """计算基于 MAE 的奖励 (normalized Reward = 1 / (1 + MAE))。

        Shapes:
            pred_pseudobulks: [K, M, G]
            real_pseudobulks: [M, G]
        Returns:
            mae: [K,M]
            mae_reward: [K, M]
        """  
        # pred_pseudobulks [K, M, G]    real_pseudobulks [M, G] -> [1, M, G]
        mae = F.l1_loss(pred_pseudobulks, real_pseudobulks.unsqueeze(0), reduction="none").mean(dim=-1)
        return mae,1.0 / (1 + mae + 1e-8)  # [K,M]

    def calculate_pds_reward(
        self,
        delta_pred_pseudobulks: torch.Tensor,  # [K, M, G]
        delta_real_pseudobulks: torch.Tensor,  # [M, G]
    ) -> torch.Tensor:
        """按照官方PDS (Perturbation Discrimination Score) 逻辑计算奖励
        Shapes:
            delta_pred_pseudobulks: [K, M, G]
            delta_real_pseudobulks: [M, G]
        Returns:
            pds_reward: [K, M]
        """
        K, M, G = delta_pred_pseudobulks.shape
        
        # 只计算对角线距离（预测第m个与真实第m个的距离）
        target_distances = (delta_pred_pseudobulks - delta_real_pseudobulks.unsqueeze(0)).abs().sum(dim=-1)  # [K, M]
        
        # 计算每个预测与所有真实的距离，用于排名
        all_distances = (
            delta_pred_pseudobulks.unsqueeze(2) - 
            delta_real_pseudobulks.unsqueeze(0).unsqueeze(0)
        ).abs().sum(dim=-1)  # [K, M, M]
        
        # 计算有多少距离小于目标距离（即排名）
        ranks = (all_distances < target_distances.unsqueeze(-1)).sum(dim=-1)  # [K, M]
        
        return 1.0 - (ranks.float() / M) # [K, M]


    def calculate_des_reward(
        self,
        delta_pred_pseudobulks: torch.Tensor,  # [K, M, G]
        delta_real_pseudobulks: torch.Tensor,  # [M, G]
    ) -> torch.Tensor:
        """
        取%k_percentage的基因 计算更精确的 DES 代理奖励。

        Shapes:
            delta_pred_pseudobulks: [K, M, G]
            delta_real_pseudobulks: [M, G]
        Returns:
            des_reward: [K, M]
        """
        K, M, G = delta_pred_pseudobulks.shape
        k_val = int(G * self.k_percentage)
        if k_val == 0:
            return torch.zeros((K, M), device=delta_pred_pseudobulks.device)

        # 确定真实的 Top-K 基因索引
        _, topk_real_indices = torch.topk(
            delta_real_pseudobulks.abs(), k=k_val, dim=-1
        )  # [M, k_val]

        # 找到预测的 Top-K 基因索引
        _, topk_pred_indices = torch.topk(
            delta_pred_pseudobulks.abs(), k=k_val, dim=-1
        )  # [K, M, k_val]

        # 计算重叠率
        # 我们将 real_indices 扩展维度，以便和 pred_indices 比较
        # [M, k_val] -> [1, M, k_val, 1]
        expanded_real = topk_real_indices.unsqueeze(0).unsqueeze(3)
        # [K, M, k_val] -> [K, M, 1, k_val]
        expanded_pred = topk_pred_indices.unsqueeze(2)

        # 利用 broadcasting 生成一个 [K, M, k_val, k_val] 的布尔矩阵
        # 矩阵中的 (i, j) 元素为 True 表示 real 的第 i 个 top 基因和 pred 的第 j 个 top 基因是同一个
        matches_matrix = expanded_real == expanded_pred

        # 沿着最后一个维度求和，得到每个 pred top 基因是否在 real top 基因列表中
        # 然后沿着倒数第二个维度求和，得到总的重合数量
        overlap_counts = matches_matrix.any(dim=2).sum(dim=2)  # [K, M]

        # 最终奖励 = 交集大小 / K
        reward = overlap_counts.float() / k_val
        return reward
 
    @torch.no_grad()
    def calculate_des_reward_hpdex(
        self,
        y_pred: torch.Tensor,         # [K, M, G] 模型预测
        basal_cells: torch.Tensor,   # [B, L, G] 对照组
        pert_names: List[str],        # [M] 扰动名称
        num_workers: int = 8,
        fdr_threshold: float = 0.05
    ) -> Dict[str, float]:
        """
        在验证集上计算DES分数 (K=1的情况)
        
        Returns:
            Dict包含每个扰动的DES分数和总体平均DES
        """
        K, M, G = y_pred.shape
        eps = 1e-8

        pred_expressions = y_pred.reshape(-1, G) .cpu().numpy()     # [M*K, G]
        control_expressions = basal_cells.reshape(-1, G).cpu().numpy()  # [B*L, G]
        
        var_df = pd.DataFrame(index=self.gene_list)
               
        pred_group_labels = K*pert_names + ['control'] * len(control_expressions)
        pred_obs_df = pd.DataFrame({'perturbation': pred_group_labels})
        pred_all_X = np.vstack([pred_expressions, control_expressions])
        adata_pred = ad.AnnData(X=pred_all_X, obs=pred_obs_df, var=var_df)
        
        pred_de_results = parallel_differential_expression(
            adata_pred,
            groupby_key="perturbation",
            reference="control", 
            num_workers=num_workers,
            min_samples=1,
        )
        
        # 计算每个扰动的DES分数
        unique_perts = list(set(pert_names))
        des_scores = {}
        
        for pert in unique_perts:
            real_pert_results = self.de_results[self.de_results['target'] == pert]
            G_true = set(real_pert_results[real_pert_results['fdr'] < fdr_threshold]['feature'])
            n_true = len(G_true)
            
            pred_pert_results = pred_de_results[pred_de_results['target'] == pert]
            G_pred = set(pred_pert_results[pred_pert_results['fdr'] < fdr_threshold]['feature'])
            n_pred = len(G_pred)
            
            if n_true == 0:
                des_scores[pert] = 0.0
                continue
        
            if n_pred <= n_true:
                # Case 1: n_pred <= n_true
                intersection_size = len(G_pred.intersection(G_true))
                des_score = intersection_size / (n_true + eps)
            else:
                # Case 2: n_pred > n_true
                # 选择fold change绝对值最大的n_true个基因
                significant_pred_df = pred_pert_results[
                    pred_pert_results['feature'].isin(G_pred)
                ].copy()
                
                significant_pred_df['abs_log2_fold_change'] = \
                    significant_pred_df['log2_fold_change'].abs()
                
                # 按照绝对fold change降序排列，取前n_true个
                sorted_df = significant_pred_df.sort_values(
                    'abs_log2_fold_change', ascending=False
                )
                G_pred_tilde = set(sorted_df.head(n_true)['feature'])
                
                intersection_size = len(G_pred_tilde.intersection(G_true))
                des_score = intersection_size / (n_true + eps)
            
            des_scores[pert] = des_score
            
            #print(f"{pert}: n_true={n_true}, n_pred={n_pred}, DES={des_score:.4f}")
        
        mean_des = np.mean(list(des_scores.values())) if des_scores else 0.0
        
        return mean_des, des_scores # 标量,字典
