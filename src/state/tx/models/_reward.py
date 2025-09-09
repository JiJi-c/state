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
import time

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

        self.gene_to_idx = {gene: i for i, gene in enumerate(self.gene_list)}
        self.pert_to_idx = {pert: i for i, pert in enumerate(self.pert_list)}
        self.idx_to_pert = {i: pert for i, pert in enumerate(self.pert_list)}

    def aggregate_rewards(
        self,
        y_samples: torch.Tensor,
        y_real: torch.Tensor,
        basal_cells: torch.Tensor,
        pert_names: List[str],
    ) -> torch.Tensor:
        """
        计算并组合奖励，返回每个细胞样本的最终奖励值。
        """
        K, N, G = y_samples.shape

        # 准备索引信息
        pert_indices_global = torch.tensor(
            [self.pert_to_idx.get(name, -1) for name in pert_names], # 使用-1标记未知扰动
            device=y_samples.device,
            dtype=torch.long,
        )
        
        # 过滤掉不在pert_list中的扰动
        valid_mask = pert_indices_global != -1
        pert_indices_global = pert_indices_global[valid_mask]
        y_samples = y_samples[:, valid_mask, :]
        y_real = y_real[valid_mask, :]
        basal_cells = basal_cells[valid_mask, :]
        N = y_samples.shape[1]

        local_perts_global, local_indices = pert_indices_global.unique(sorted=True, return_inverse=True)
        M = int(local_perts_global.numel())
        pert_indices = local_indices # [N_valid]
        
        # 计算Pseudobulks
        delta_real = y_real - basal_cells
        delta_pred = y_samples - basal_cells.unsqueeze(0)
        
        delta_real_pseudobulks = self._vectorized_group_mean(
            delta_real.unsqueeze(0), pert_indices, M
        ).squeeze(0)  # [M, G]
        delta_pred_pseudobulks = self._vectorized_group_mean(
            delta_pred, pert_indices, M
        )  # [K, M, G]

        # PDS: 向量化目标基因置零
        pert_names_in_batch = [self.idx_to_pert.get(idx.item()) for idx in local_perts_global]
        
        # 找到需要置零的基因索引和它们在伪批量中的位置
        pert_indices_in_M = []
        gene_indices_to_zero = []
        for i, name in enumerate(pert_names_in_batch):
            if name in self.gene_to_idx:
                pert_indices_in_M.append(i)
                gene_indices_to_zero.append(self.gene_to_idx[name])
        
        pds_delta_pred = delta_pred_pseudobulks.clone()
        pds_delta_real = delta_real_pseudobulks.clone()

        if pert_indices_in_M: 
            pert_indices_in_M = torch.tensor(pert_indices_in_M, device=y_samples.device, dtype=torch.long)
            gene_indices_to_zero = torch.tensor(gene_indices_to_zero, device=y_samples.device, dtype=torch.long)

            pds_delta_real[pert_indices_in_M, gene_indices_to_zero] = 0
            pds_delta_pred[:, pert_indices_in_M, gene_indices_to_zero] = 0


        pds_r = self.calculate_pds_reward(pds_delta_pred, pds_delta_real)
            

        # des_r = self.calculate_des_reward(delta_pred_pseudobulks, delta_real_pseudobulks)
        des_r = self.calculate_pearson_reward(delta_pred_pseudobulks, delta_real_pseudobulks)


        # 组合奖励
        pds_w = self.reward_weights.get("pds", 1.0)
        des_w = self.reward_weights.get("des", 1.0)

        total_weight = pds_w + des_w + 1e-8 
        total_rewards_per_pert = (pds_w * pds_r + des_w * des_r) / total_weight # [K, M]
        
        # 每一个扰动归一化奖励
        # mean_R_pert = total_rewards_per_pert.mean(dim=0, keepdim=True)
        # std_R_pert = total_rewards_per_pert.std(dim=0, keepdim=True)
        # total_rewards_per_pert = (total_rewards_per_pert - mean_R_pert) / (std_R_pert + 1e-8)
            
        # 将奖励映射回每个细胞
        final_rewards = total_rewards_per_pert[:, pert_indices] # [K, N_valid]
        
        return final_rewards, pds_r, des_r, M


    def aggregate_rewards_hpdex(
        self,
        y_samples: torch.Tensor,
        y_real: torch.Tensor,
        basal_cells: torch.Tensor,
        pert_names: List[str],
    ) -> torch.Tensor:
        """
        [Experimental] 计算并组合奖励，使用 hpdex 计算精确的DES分数。
        """
        K, N, G = y_samples.shape

        # --- 1. 准备索引信息 ---
        pert_indices_global = torch.tensor(
            [self.pert_to_idx.get(name, -1) for name in pert_names], # 使用-1标记未知扰动
            device=y_samples.device,
            dtype=torch.long,
        )
        
        # 过滤掉不在pert_list中的扰动
        valid_mask = pert_indices_global != -1
        pert_indices_global = pert_indices_global[valid_mask]
        y_samples_valid = y_samples[:, valid_mask, :]
        y_real_valid = y_real[valid_mask, :]
        basal_cells_valid = basal_cells[valid_mask, :]
        pert_names_valid = [name for i, name in enumerate(pert_names) if valid_mask[i]]

        local_perts_global, local_indices = pert_indices_global.unique(sorted=True, return_inverse=True)
        M = int(local_perts_global.numel())
        pert_indices = local_indices # [N_valid]
        
        # --- 2. 为PDS计算伪批量 ---
        delta_real = y_real_valid - basal_cells_valid
        delta_pred = y_samples_valid - basal_cells_valid.unsqueeze(0)
        
        delta_real_pseudobulks = self._vectorized_group_mean(
            delta_real.unsqueeze(0), pert_indices, M
        ).squeeze(0)
        delta_pred_pseudobulks = self._vectorized_group_mean(
            delta_pred, pert_indices, M
        )

        # --- 3. PDS: 向量化目标基因置零 ---
        pert_names_in_batch = [self.idx_to_pert.get(idx.item()) for idx in local_perts_global]
        pert_indices_in_M, gene_indices_to_zero = [], []
        for i, name in enumerate(pert_names_in_batch):
            if name in self.gene_to_idx:
                pert_indices_in_M.append(i)
                gene_indices_to_zero.append(self.gene_to_idx[name])
        
        pds_delta_pred = delta_pred_pseudobulks.clone()
        pds_delta_real = delta_real_pseudobulks.clone()

        if pert_indices_in_M:
            pert_indices_in_M = torch.tensor(pert_indices_in_M, device=y_samples.device, dtype=torch.long)
            gene_indices_to_zero = torch.tensor(gene_indices_to_zero, device=y_samples.device, dtype=torch.long)
            pds_delta_real[pert_indices_in_M, gene_indices_to_zero] = 0
            pds_delta_pred[:, pert_indices_in_M, gene_indices_to_zero] = 0

        # --- 4. 计算各项奖励 ---
        pds_r = self.calculate_pds_reward(pds_delta_pred, pds_delta_real)
        
        # 调用 hpdex DES 函数
        # 假设 `basal_cells` 是用于DE分析的对照组细胞池
        start_time = time.time()
        des_r = self.calculate_des_reward_hpdex(
            y_samples=y_samples_valid,
            y_real=y_real_valid,
            control_pool=basal_cells_valid, # 使用完整的、未经过滤的basal_cells作为对照组
            pert_names=pert_names_valid,
            local_perts_global=local_perts_global,
            num_workers=64,
        )
        end_time = time.time()
        print(f"hpdex DES time: {end_time - start_time:.2f} seconds")

        # --- 5. 组合奖励 ---
        pds_w = self.reward_weights.get("pds", 1.0)
        des_w = self.reward_weights.get("des", 1.0)
        total_weight = pds_w + des_w + 1e-8 
        total_rewards_per_pert = (pds_w * pds_r + des_w * des_r) / total_weight
        
        mean_R_pert = total_rewards_per_pert.mean(dim=0, keepdim=True)
        std_R_pert = total_rewards_per_pert.std(dim=0, keepdim=True)
        total_rewards_per_pert = (total_rewards_per_pert - mean_R_pert) / (std_R_pert + 1e-8)

        final_rewards = total_rewards_per_pert[:, pert_indices]

        return final_rewards, pds_r, des_r, M


    def _vectorized_group_mean(
        self, values: torch.Tensor, indices: torch.Tensor, num_groups: int
    ) -> torch.Tensor:
        """使用纯PyTorch高效地、向量化地实现按索引分组求均值。

        Shapes:
            values:  [K, N, G]
            indices: [N]  # 取值范围 [0, num_groups)
            num_groups: int  # = M

        Returns:
            group_means: [K, M, G]
        """
        K_dim, N_dim, G_dim = values.shape

        # 将 K 和 N 维度展平，以便进行分组操作
        values_flat = values.reshape(-1, G_dim) # N*K, G
        indices_flat = indices.repeat(K_dim) # N*K
    
        offsets = (
            torch.arange(K_dim, device=values.device, dtype=indices.dtype) * num_groups
        )
        indices_offset = indices_flat + offsets.unsqueeze(1).expand(-1, N_dim).reshape(-1)
        
        total_groups = K_dim * num_groups

        summed_values = torch.zeros(
            total_groups, G_dim, device=values.device, dtype=values.dtype
        )
        summed_values.scatter_add_(
            0, indices_offset.unsqueeze(1).expand_as(values_flat), values_flat
        )

        counts = torch.zeros(total_groups, device=values.device, dtype=torch.long)
        counts.scatter_add_(
            0, indices_offset, torch.ones_like(indices_offset, dtype=torch.long)
        )

        mean_values = summed_values / counts.unsqueeze(1).clamp(min=1)

        # 重塑为 [K, M, G]
        return mean_values.reshape(K_dim, num_groups, G_dim)


    def calculate_mae_reward(
        self, pred_pseudobulks: torch.Tensor, real_pseudobulks: torch.Tensor
    ) -> torch.Tensor:
        """计算基于 MAE 的奖励 (normalized Reward = 1 / (1 + MAE))。

        Shapes:
            pred_pseudobulks: [K, M, G]
            real_pseudobulks: [M, G]
        Returns:
            mae_reward: [K, M]
        """
        # real_pseudobulks [M, G] -> [1, M, G]
        real_expanded = real_pseudobulks.unsqueeze(0)
        # pred_pseudobulks [K, M, G]
        mae = F.l1_loss(pred_pseudobulks, real_expanded, reduction="none").mean(dim=-1)
        return 1.0 / (1 + mae + 1e-8)  # [K,M]


    def calculate_pds_reward(
        self,
        delta_pred_pseudobulks: torch.Tensor,  # [K, M, G]
        delta_real_pseudobulks: torch.Tensor,  # [M, G]
    ) -> torch.Tensor:
        """按照官方PDS (Perturbation Discrimination Score) 逻辑计算奖励
        核心区别:
        1. 使用扰动效应的绝对值 `abs(delta)` 进行比较.
        2. 采用官方的归一化排名公式 `1 - rank / M` (0-based rank).

        Shapes:
            delta_pred_pseudobulks: [K, M, G]
            delta_real_pseudobulks: [M, G]
        Returns:
            pds_reward: [K, M]
        """
        K, M, G = delta_pred_pseudobulks.shape

        actual_perts_mask = (delta_real_pseudobulks.abs().sum(dim=-1) > 0)  # [M]
        num_actual_perts = actual_perts_mask.sum().item()
        
        if num_actual_perts < self.min_perts_for_pds:
            return torch.zeros(K, M, device=delta_pred_pseudobulks.device)

        # 使用效应绝对值
        pred_effects = delta_pred_pseudobulks.abs()
        real_effects = delta_real_pseudobulks.abs()

        # 步骤2: 计算L1距离矩阵
        pred_expanded = pred_effects.unsqueeze(2)  # [K, M, 1, G]
        real_expanded = real_effects.unsqueeze(0).unsqueeze(0)  # [1, 1, M, G]
        distance_matrix = (pred_expanded - real_expanded).abs().sum(dim=-1)  # [K, M, M]

        # 向量化计算排名
        # `sorted_indices` will contain, for each predicted effect (dim 1),
        # the indices of the real effects sorted by distance.
        sorted_indices = torch.argsort(distance_matrix, dim=-1)  # [K, M, M]

        # We need to find the rank of the *correct* real effect for each prediction.
        # For the m-th prediction, the correct real effect is also at index m.
        # So we create a target tensor of shape [1, M, 1] containing [0, 1, ..., M-1]
        m_indices = torch.arange(M, device=distance_matrix.device).view(1, M, 1)

        # `sorted_indices == m_indices` will broadcast to a boolean mask [K, M, M].
        # For each [k, m, :], there will be one `True` at the position (rank) of `m`.
        comparison_mask = (sorted_indices == m_indices)

        # `argmax` finds the index of the first `True`, which is the 0-based rank.
        rank_positions = torch.argmax(comparison_mask.int(), dim=-1)  # [K, M]

        # 计算归一化排名分数, 使用官方公式 1 - rank / M (0-based rank)
        norm_ranks = 1.0 - (rank_positions.float() / M) # [K, M]

        return norm_ranks


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

    def calculate_pearson_reward(
        self,
        delta_pred_pseudobulks: torch.Tensor,  # [K, M, G]
        delta_real_pseudobulks: torch.Tensor,  # [M, G]
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        [New Proxy]
        使用预测扰动效应和真实扰动效应之间的皮尔逊相关系数计算代理DES奖励。
        此方法计算速度快，且能很好地反映基因表达变化趋势的相似性。

        Shapes:
            delta_pred_pseudobulks: [K, M, G]
            delta_real_pseudobulks: [M, G]
        Returns:
            pearson_reward: [K, M]
        """
        # 展开 real 以便广播
        real_expanded = delta_real_pseudobulks.unsqueeze(0)  # [1, M, G]

        # 减去均值 (Center the vectors)
        pred_centered = delta_pred_pseudobulks - delta_pred_pseudobulks.mean(dim=-1, keepdim=True)
        real_centered = real_expanded - real_expanded.mean(dim=-1, keepdim=True)

        # 计算协方差的分子部分 (sum of products)
        numerator = (pred_centered * real_centered).sum(dim=-1)

        # 计算标准差的分子部分 (product of sum of squares)
        pred_denom = torch.sqrt((pred_centered**2).sum(dim=-1))
        real_denom = torch.sqrt((real_centered**2).sum(dim=-1))
        denominator = pred_denom * real_denom + eps

        # 计算皮尔逊相关系数
        pearson_corr = numerator / denominator  # [K, M]

        # 将结果从 [-1, 1] 映射到 [0, 1] 作为奖励
        reward = (pearson_corr + 1) / 2.0
        
        return reward

    def calculate_des_reward_original(
        self,
        y_samples: torch.Tensor,       # [K, N, G]
        y_real: torch.Tensor,          # [N, G]
        basal_cells: torch.Tensor,     # [N, G]
        pert_names: List[str],
    ) -> List[Dict[str, float]]: # <--- 返回类型已更改
        """
        使用 scanpy 为 K 个采样中的每一个独立计算 DES 分数。
        """
        K = y_samples.shape[0]
        
        # 数据准备
        real_expressions = y_real.cpu().numpy()
        basal_expressions = basal_cells.cpu().numpy()
        
        # 'control' 作为DE分析的参照组
        group_labels = pert_names + ['control'] * basal_expressions.shape[0]
        var_df = pd.DataFrame(index=self.gene_list)
        obs_df = pd.DataFrame({'perturbation': group_labels})

        # 对真实数据只进行一次DE分析
        all_real_X = np.vstack([real_expressions, basal_expressions])
        adata_real = ad.AnnData(X=all_real_X, obs=obs_df, var=var_df)
        sc.tl.rank_genes_groups(adata_real, groupby='perturbation', reference='control', method='wilcoxon', key_added="rank_genes_real")

        all_k_des_scores = []
        unique_perts = [p for p in np.unique(pert_names) if p != 'control']

        # 循环遍历 K 个采样
        for k in range(K):
            pred_expressions = y_samples[k, :, :].cpu().numpy()
            all_pred_X = np.vstack([pred_expressions, basal_expressions])
            adata_pred = ad.AnnData(X=all_pred_X, obs=obs_df, var=var_df)

            sc.tl.rank_genes_groups(adata_pred, groupby='perturbation', reference='control', method='wilcoxon', key_added="rank_genes_pred")

            des_scores_for_k = {}
            for pert in unique_perts:
                real_de_results = sc.get.rank_genes_groups_df(adata_real, group=pert, key="rank_genes_real")
                pred_de_results = sc.get.rank_genes_groups_df(adata_pred, group=pert, key="rank_genes_pred")
                
                # DES 计算
                fdr_threshold = 0.05
                g_true_sig = set(real_de_results[real_de_results['pvals_adj'] < fdr_threshold]['names'])
                g_pred_sig = set(pred_de_results[pred_de_results['pvals_adj'] < fdr_threshold]['names'])
                n_true, n_pred = len(g_true_sig), len(g_pred_sig)
                score = 0.0
                if n_true > 0:
                    if n_pred <= n_true:
                        intersection_size = len(g_pred_sig.intersection(g_true_sig))
                        score = intersection_size / n_true
                    else:
                        pred_de_results['abs_logfoldchanges'] = pred_de_results['logfoldchanges'].abs()
                        pred_de_results_sorted = pred_de_results.sort_values('abs_logfoldchanges', ascending=False)
                        significant_pred_genes = pred_de_results_sorted[pred_de_results_sorted['names'].isin(g_pred_sig)]
                        g_pred_tilde = set(significant_pred_genes.head(n_true)['names'])
                        intersection_size = len(g_pred_tilde.intersection(g_true_sig))
                        score = intersection_size / n_true
                des_scores_for_k[pert] = score
            
            all_k_des_scores.append(des_scores_for_k)

        return all_k_des_scores

    def calculate_des_reward_pytorch(
        self,
        y_samples: Tensor,       # [K, N, G]
        y_real: Tensor,          # [N, G]
        basal_cells: Tensor,     # [N, G]
        pert_indices: Tensor,    # [N]
        M: int,                  # 批次中的唯一扰动数
    ) -> Tensor:
        """
        [Final Version]
        使用纯PyTorch在GPU上计算精确的DES分数。
        该版本在基因维度上完全向量化，并在扰动和采样维度上使用清晰的循环。
        """
        K, N, G = y_samples.shape
        fdr_threshold = 0.05
        eps = 1e-8

        pert_mask = F.one_hot(pert_indices, num_classes=M).T.bool() # [M, N]

        # --- 1. 对真实数据执行DE分析 ---
        real_de_p_values = torch.ones(M, G, device=y_samples.device)
        for i in range(M):
            if pert_mask[i].any():
                pert_cells_real = y_real[pert_mask[i]]
                real_de_p_values[i] = self._wilcoxon_rank_sums_core(pert_cells_real, basal_cells)
        
        real_de_fdr = self._vectorized_bh_correction(real_de_p_values)
        g_true_sig = real_de_fdr < fdr_threshold # [M, G]
        n_true = g_true_sig.sum(dim=1).float()   # [M]

        # --- 2. 对 K 个预测样本执行DE分析 (使用分块策略避免OOM) ---
        pred_de_p_values = torch.ones(K, M, G, device=y_samples.device)
        
        # 如果K=8, chunk_size=2，则峰值内存约为原来的 1/4
        k_chunk_size = 4
        
        for k_start in range(0, K, k_chunk_size):
            k_end = min(k_start + k_chunk_size, K)
            y_samples_chunk = y_samples[k_start:k_end]
            
            for i in range(M):
                if pert_mask[i].any():
                    # group1_batch shape: [k_chunk_size, num_cells_for_pert_i, G]
                    pert_cells_pred_batch = y_samples_chunk[:, pert_mask[i], :]
                    
                    # Batched Wilcoxon call, p-values shape [k_chunk_size, G]
                    p_vals_chunk = self._wilcoxon_rank_sums_batched(
                        pert_cells_pred_batch, basal_cells
                    )
                    
                    # 将块的结果放回总张量
                    pred_de_p_values[k_start:k_end, i, :] = p_vals_chunk

        pred_de_fdr = self._vectorized_bh_correction(pred_de_p_values)
        g_pred_sig = pred_de_fdr < fdr_threshold # [K, M, G]
        n_pred = g_pred_sig.sum(dim=2).float()   # [K, M]
        
        # --- 3. 计算DES分数 ---
        n_true_expanded = n_true.unsqueeze(0).expand(K, -1)
        intersection_size = (g_pred_sig & g_true_sig.unsqueeze(0)).sum(dim=2).float()
        score_case1 = intersection_size / (n_true_expanded + eps)

        # 严格计算 log2 Fold Change 用于排序
        y_samples_counts = torch.expm1(y_samples)
        y_real_counts = torch.expm1(y_real)
        basal_cells_counts = torch.expm1(basal_cells)
        control_mean_counts = basal_cells_counts.mean(dim=0)
        
        n_pert_per_group = pert_mask.sum(dim=1).float()
        pert_mask_float = pert_mask.float()
        real_mean_counts = torch.einsum('mn,ng->mg', pert_mask_float, y_real_counts) / (n_pert_per_group.view(-1, 1) + eps)
        pred_mean_counts = torch.einsum('mn,kng->kmg', pert_mask_float, y_samples_counts) / (n_pert_per_group.view(1, -1, 1) + eps)
        
        pred_log2fc = torch.log2(pred_mean_counts / (control_mean_counts.view(1, 1, -1) + eps) + eps)
        pred_effects = pred_log2fc.abs()
        
        pred_effects[~g_pred_sig] = -1
        _, sorted_indices = torch.sort(pred_effects, dim=2, descending=True)
        ranks = torch.zeros_like(sorted_indices)
        ranks.scatter_(2, sorted_indices, torch.arange(G, device=y_samples.device).expand_as(sorted_indices))
        
        n_true_int = n_true.long()
        mask_top_k = ranks < n_true_int.unsqueeze(0).unsqueeze(2)
        g_pred_tilde = g_pred_sig & mask_top_k
        intersection_size_case2 = (g_pred_tilde & g_true_sig.unsqueeze(0)).sum(dim=2).float()
        score_case2 = intersection_size_case2 / (n_true_expanded + eps)

        condition = n_pred <= n_true_expanded
        des_reward = torch.where(condition, score_case1, score_case2)
        des_reward.masked_fill_(n_true_expanded == 0, 0.0)

        return des_reward

    def _wilcoxon_rank_sums_batched(
        self, group1_batch: Tensor, group2: Tensor, eps: float = 1e-8
    ) -> Tensor:
        """
        [Batched Version]
        纯PyTorch实现的、在批次维度(B)上完全向量化的Wilcoxon秩和检验。
        用于一次性处理 K 个预测样本。
        Shapes:
            group1_batch: [B, n1, G]
            group2:       [n2, G]
        Returns:
            p_values:     [B, G]
        """
        B, n1, G = group1_batch.shape
        n2, _ = group2.shape
        n_total = n1 + n2

        if n1 == 0 or n2 == 0:
            return torch.ones(B, G, device=group1_batch.device)

        # 1. 批次化合并与计算秩次
        group2_expanded = group2.unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([group1_batch, group2_expanded], dim=1)  # [B, n_total, G]

        # 转置为 (B, G, N) 以便使用批次兼容的searchsorted
        combined_t = combined.transpose(1, 2)
        sorted_vals_t, _ = torch.sort(combined_t, dim=-1)
        ranks_lower_t = torch.searchsorted(sorted_vals_t, combined_t, side="left")
        ranks_upper_t = torch.searchsorted(sorted_vals_t, combined_t, side="right")
        ranks_t = (ranks_lower_t + ranks_upper_t).float() / 2.0 + 1
        ranks = ranks_t.transpose(1, 2)  # 转回 (B, N, G)

        # 2. 批次化计算U统计量
        rank_sum_group1 = ranks[:, :n1, :].sum(dim=1)  # [B, G]
        u_statistic = rank_sum_group1 - (n1 * (n1 + 1)) / 2

        # 3. 批次化计算Z分数 (包含完全向量化的重复值方差修正)
        mean_u = n1 * n2 / 2

        # 创建批次和基因的ID，用于定位每个值
        batch_ids = torch.arange(B, device=combined.device).view(B, 1, 1).expand(-1, n_total, G)
        gene_ids = torch.arange(G, device=combined.device).view(1, 1, G).expand(B, n_total, -1)
        
        # 注意: 此处会产生一个巨大的中间张量 (B*N*G, 3)，可能消耗大量VRAM。
        # 这是为了极致性能，用空间换时间的策略。
        triplets = torch.stack([
            batch_ids.flatten(),
            gene_ids.flatten(),
            combined.flatten()
        ], dim=1)

        unique_triplets, counts = torch.unique(triplets, dim=0, return_counts=True)

        tie_mask = counts > 1
        tie_counts = counts[tie_mask]
        
        # 从唯一值中提取批次和基因ID
        tie_batch_ids = unique_triplets[tie_mask, 0].long()
        tie_gene_ids = unique_triplets[tie_mask, 1].long()
        
        correction_terms = tie_counts.pow(3) - tie_counts

        # 使用1D索引将修正项高效地加回到对应的 (batch, gene) 位置
        flat_index = tie_batch_ids * G + tie_gene_ids
        tie_correction_flat = torch.zeros(B * G, device=combined.device, dtype=correction_terms.dtype)
        tie_correction_flat.scatter_add_(0, flat_index, correction_terms)
        tie_correction = tie_correction_flat.view(B, G)

        var_u = (n1 * n2 / (n_total * (n_total - 1) + eps)) * \
                (((n_total**3 - n_total) / 12.0) - (tie_correction / 12.0))

        std_u = torch.sqrt(var_u.clamp(min=0))
        z_score = (u_statistic - mean_u) / (std_u + eps)

        # 4. 计算双尾p值
        normal_dist = torch.distributions.Normal(0, 1)
        p_values = 2 * (1 - normal_dist.cdf(torch.abs(z_score)))

        return p_values

    def _wilcoxon_rank_sums_core(self, group1: Tensor, group2: Tensor, eps: float = 1e-8) -> Tensor:
        """
        [Fully Vectorized Core for single comparison]
        纯PyTorch实现的、处理单组比较的Wilcoxon秩和检验。
        包含对重复值的、完全向量化的方差修正。
        """
        n1, G = group1.shape
        n2, _ = group2.shape
        n_total = n1 + n2

        if n1 == 0 or n2 == 0:
            return torch.ones(G, device=group1.device)

        # 1. 合并与计算秩次 (处理重复值)
        combined = torch.cat([group1, group2], dim=0)
        sorted_vals, _ = torch.sort(combined, dim=0)
        
        ranks_lower = torch.searchsorted(sorted_vals.T, combined.T, side='left').T
        ranks_upper = torch.searchsorted(sorted_vals.T, combined.T, side='right').T
        ranks = (ranks_lower + ranks_upper).float() / 2.0 + 1

        # 2. 计算U统计量
        rank_sum_group1 = ranks[:n1, :].sum(dim=0)
        u_statistic = rank_sum_group1 - (n1 * (n1 + 1)) / 2
        
        # 3. 计算Z分数 (包含完全向量化的重复值方差修正)
        mean_u = n1 * n2 / 2
        
        # --- [新] 完全向量化的重复值方差修正 ---
        # 1. 创建 (gene_id, value) 对的扁平化张量
        gene_ids = torch.arange(G, device=combined.device).expand(n_total, -1)

        # 2. 将 gene_id 和 value 堆叠成对。由于是行优先扁平化，配对是正确的。
        pairs = torch.stack([gene_ids.flatten(), combined.flatten()], dim=1)

        # 3. 获取唯一的 (gene, value) 对及其计数
        unique_pairs, counts = torch.unique(pairs, dim=0, return_counts=True)

        # 4. 识别出现次数大于1的为重复值 (ties)
        tie_mask = counts > 1
        tie_counts = counts[tie_mask]

        # 5. 获取每个重复值组对应的基因ID
        tie_gene_ids = unique_pairs[tie_mask, 0].long()

        # 6. 计算每个重复值组的修正项 t^3 - t
        correction_terms = tie_counts.pow(3) - tie_counts

        # 7. 使用 scatter_add_ 高效地将修正项按基因ID求和
        tie_correction = torch.zeros(G, device=combined.device, dtype=correction_terms.dtype)
        tie_correction.scatter_add_(0, tie_gene_ids, correction_terms)
            
        var_u = (n1 * n2 / (n_total * (n_total - 1) + eps)) * \
                (((n_total**3 - n_total) / 12.0) - (tie_correction / 12.0))
        
        std_u = torch.sqrt(var_u.clamp(min=0))
        z_score = (u_statistic - mean_u) / (std_u + eps)

        # 4. 计算双尾p值
        normal_dist = torch.distributions.Normal(0, 1)
        p_values = 2 * (1 - normal_dist.cdf(torch.abs(z_score)))
        
        return p_values

    def _vectorized_bh_correction(self, p_values: Tensor, eps: float = 1e-8) -> Tensor:
        """纯PyTorch实现的向量化Benjamini-Hochberg FDR校正"""
        shape = p_values.shape
        p_values_flat = p_values.view(-1)
        G = len(p_values_flat)

        sorted_p, sort_indices = torch.sort(p_values_flat)
        
        ascending_ranks = torch.arange(1, G + 1, device=p_values.device, dtype=torch.float32)
        raw_q_sorted = sorted_p * G / ascending_ranks
        
        # 使用clamp确保q值不会超过1
        q_values_sorted = torch.clamp(
            torch.cummin(raw_q_sorted.flip(dims=[0]), dim=0)[0].flip(dims=[0]),
            min=0, max=1
        )
        
        fdr_values_flat = torch.empty_like(q_values_sorted)
        fdr_values_flat.scatter_(0, sort_indices, q_values_sorted)

        return fdr_values_flat.view(shape)


    def calculate_des_reward_hpdex(
        self,
        y_samples: torch.Tensor,      # [K, N_valid, G]
        y_real: torch.Tensor,         # [N_valid, G]
        control_pool: torch.Tensor,   # [N_control, G]
        pert_names: List[str],        # [N_valid]
        local_perts_global: torch.Tensor, # [M]
        num_workers: int = 64,
    ) -> torch.Tensor:
        """
        使用 hpdex 在CPU上计算DES
        """
        K, N, G = y_samples.shape
        M = len(local_perts_global)
        fdr_threshold = 0.05
        eps = 1e-8

        #1. 数据准备 (CPU转换)
        real_expressions = y_real.cpu().numpy()
        control_expressions = control_pool.cpu().numpy()
        
        group_labels = pert_names + ['control'] * control_expressions.shape[0]
        var_df = pd.DataFrame(index=self.gene_list)
        obs_df = pd.DataFrame({'perturbation': group_labels})

        #2. 真实数据的DE分析 
        all_real_X = np.vstack([real_expressions, control_expressions])
        adata_real = ad.AnnData(X=all_real_X, obs=obs_df, var=var_df)
        
        real_de_results = parallel_differential_expression(
            adata_real, 
            groupby_key="perturbation",
            reference="control",
            num_workers=num_workers
        )
        
        pert_names_in_batch = [self.idx_to_pert.get(idx.item()) for idx in local_perts_global]
        g_true_sig, n_true = {}, {}
        for pert in pert_names_in_batch:
            pert_results = real_de_results[real_de_results['target'] == pert]
            g_true_sig[pert] = set(pert_results[pert_results['fdr'] < fdr_threshold]['feature'])
            n_true[pert] = len(g_true_sig[pert])

        #3. 循环遍历 K 个采样 
        all_k_des_scores = torch.zeros(K, M, device=y_samples.device, dtype=torch.float32)

        for k in range(K):
            pred_expressions = y_samples[k, :, :].cpu().numpy()
            all_pred_X = np.vstack([pred_expressions, control_expressions])
            adata_pred = ad.AnnData(X=all_pred_X, obs=obs_df, var=var_df)

            pred_de_results = parallel_differential_expression(
                adata_pred, 
                groupby_key="perturbation", 
                reference="control", 
                num_workers=num_workers
            )

            for i, pert in enumerate(pert_names_in_batch):
                n_true_pert = n_true.get(pert, 0)
                
                if n_true_pert == 0:
                    score = 0.0
                else:
                    pred_pert_results = pred_de_results[pred_de_results['target'] == pert]
                    g_pred_sig_pert = set(pred_pert_results[pred_pert_results['fdr'] < fdr_threshold]['feature'])
                    n_pred_pert = len(g_pred_sig_pert)
                    
                    if n_pred_pert <= n_true_pert:
                        intersection_size = len(g_pred_sig_pert.intersection(g_true_sig.get(pert, set())))
                        score = intersection_size / (n_true_pert + eps)
                    else:
                        significant_pred_df = pred_pert_results[pred_pert_results['feature'].isin(g_pred_sig_pert)]

                        significant_pred_df = significant_pred_df.copy()
                        significant_pred_df.loc[:, 'abs_log2_fold_change'] = significant_pred_df['log2_fold_change'].abs()
                        
                        sorted_df = significant_pred_df.sort_values('abs_log2_fold_change', ascending=False)
                        
                        g_pred_tilde = set(sorted_df.head(n_true_pert)['feature'])
                        intersection_size = len(g_pred_tilde.intersection(g_true_sig.get(pert, set())))
                        score = intersection_size / (n_true_pert + eps)
                
                all_k_des_scores[k, i] = score

        return all_k_des_scores