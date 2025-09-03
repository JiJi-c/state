import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection
import scanpy as sc
import anndata as ad

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
        """计算并组合奖励，返回每个细胞样本的最终奖励值。

        Shapes:
            y_samples: [K, N, G]  # K个采样，N个样本（一般 N = B*S），G个基因
            y_real:    [N, G]
            basal_cells: [N, G]
            batch["pert_indices"]: [N]  # 每个样本的扰动索引，取值范围 [0, M)
            num_unique_perts (M): int    # 唯一扰动的数量

        Returns:
            final_rewards: [K, N]        # 每次采样在样本维度上的奖励
            mae_r, pds_r, des_r: [K, M]  # 每种扰动的聚合奖励
        """
        pert_indices = torch.tensor(
            [self.pert_to_idx.get(name, 0) for name in pert_names],
            device=y_samples.device,
            dtype=torch.long,
        )  # [N]
        local_perts, local_indices = pert_indices.unique(sorted=True, return_inverse=True)
        M = int(local_perts.numel())
        pert_indices = local_indices 

        # 将 [N, G] → [1, N, G]，再经分组均值得到 [M, G]
        real_pseudobulks = self._vectorized_group_mean(
            y_real.unsqueeze(0), pert_indices, M
        ).squeeze(0)

        # 将 [K, N, G] 经分组均值得到 [K, M, G]
        pred_pseudobulks = self._vectorized_group_mean(
            y_samples, pert_indices, M
        )  # [K, M, G]

        # 计算 delta，用于 PDS
        delta_real = y_real - basal_cells                 # [N, G]
        delta_pred = y_samples - basal_cells.unsqueeze(0) # [K, N, G]

        # 分组均值后的 delta 伪批量
        delta_real_pseudobulks = self._vectorized_group_mean(
            delta_real.unsqueeze(0), pert_indices, M
        ).squeeze(0)  # [M, G]
        delta_pred_pseudobulks = self._vectorized_group_mean(
            delta_pred, pert_indices, M
        )  # [K, M, G]

        pert_names_in_batch = [self.idx_to_pert.get(idx.item()) for idx in local_perts]
        
        pds_delta_pred = delta_pred_pseudobulks.clone()
        pds_delta_real = delta_real_pseudobulks.clone()
        cloned = False
        for i, pert_name in enumerate(pert_names_in_batch):
            if not cloned:
                pds_delta_pred = pds_delta_pred.clone()
                pds_delta_real = pds_delta_real.clone()
                cloned = True
            
            # 检查扰动名称是否在基因列表中，如果不在则跳过
            if pert_name in self.gene_to_idx:
                target_gene_idx = self.gene_to_idx[pert_name]
                pds_delta_pred[:, i, target_gene_idx] = 0
                pds_delta_real[i, target_gene_idx] = 0


        # --- 调用新方法并处理字典列表 ---
        with torch.no_grad():
            # 1. 调用返回字典列表的新函数
            des_scores_list = self.calculate_des_reward_optimized(
                y_samples, y_real, basal_cells, pert_names
            )

        # 2. 初始化 [K, M] 奖励张量
        des_r = torch.zeros(y_samples.shape[0], M, device=y_samples.device)
        
        # 3. 遍历每个采样的分数，填充奖励张量
        for k, des_scores_dict in enumerate(des_scores_list):
            for i, pert_name in enumerate(pert_names_in_batch):
                des_r[k, i] = des_scores_dict.get(pert_name, 0.0)

        # --- 计算其他奖励 ---
        #mae_r = self.calculate_mae_reward(pred_pseudobulks, real_pseudobulks)
        pds_r = self.calculate_pds_reward(pds_delta_pred, pds_delta_real)

        #mae_w = self.reward_weights.get("mae", 0)
        pds_w = self.reward_weights.get("pds", 0)
        des_w = self.reward_weights.get("des", 0)

        total_rewards_per_pert = (
            pds_w * pds_r +
            des_w * des_r
        ) / (pds_w + des_w)

        final_rewards = total_rewards_per_pert[:, pert_indices]

        return final_rewards,  pds_r, des_r, M


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

        # 将结果重塑为 [K, M, G]
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
        2. 采用官方的归一化排名公式 `1 - (rank - 1) / (M - 1)`.


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

        # 步骤1: 使用效应绝对值
        pred_effects = delta_pred_pseudobulks.abs()
        real_effects = delta_real_pseudobulks.abs()

        # 步骤2: 计算L1距离矩阵
        pred_expanded = pred_effects.unsqueeze(2)  # [K, M, 1, G]
        real_expanded = real_effects.unsqueeze(0).unsqueeze(0)  # [1, 1, M, G]
        distance_matrix = (pred_expanded - real_expanded).abs().sum(dim=-1)  # [K, M, M]

        norm_ranks = torch.zeros(K, M, device=delta_pred_pseudobulks.device)

        for k in range(K):
            for m in range(M):
                distances = distance_matrix[k, m, :]  # [M]
                sorted_indices = torch.argsort(distances)  # [M]

                # 找到正确扰动（索引 m）在排序中的位置 (0-indexed rank)
                rank_position = (sorted_indices == m).nonzero(as_tuple=True)[0].item()

                # 步骤3: 计算归一化排名分数, 使用官方公式 1 - (rank - 1) / (total - 1)
                # 1-indexed rank is rank_position + 1
                # so, 1 - ((rank_position + 1) - 1) / (M - 1) = 1 - rank_position / (M - 1)
                score = 1.0 - (rank_position / M)
                norm_ranks[k, m] = score

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

        # --- 第一步: 确定真实的 Top-K 基因索引 ---
        _, topk_real_indices = torch.topk(
            delta_real_pseudobulks.abs(), k=k_val, dim=-1
        )  # [M, k_val]

        # --- 第二步: 找到预测的 Top-K 基因索引 ---
        _, topk_pred_indices = torch.topk(
            delta_pred_pseudobulks.abs(), k=k_val, dim=-1
        )  # [K, M, k_val]

        # --- 第三步: 计算重叠率 ---
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

    def calculate_des_reward_optimized(
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
        
        # --- 1. 数据准备 ---
        real_expressions = y_real.cpu().numpy()
        basal_expressions = basal_cells.cpu().numpy()
        
        # 'control' 将作为DE分析的参照组
        group_labels = pert_names + ['control'] * basal_expressions.shape[0]
        var_df = pd.DataFrame(index=self.gene_list)
        obs_df = pd.DataFrame({'perturbation': group_labels})

        # --- 2. 优化：对真实数据只进行一次DE分析 ---
        all_real_X = np.vstack([real_expressions, basal_expressions])
        adata_real = ad.AnnData(X=all_real_X, obs=obs_df, var=var_df)
        sc.tl.rank_genes_groups(adata_real, groupby='perturbation', reference='control', method='wilcoxon', key_added="rank_genes_real")

        all_k_des_scores = []
        unique_perts = [p for p in np.unique(pert_names) if p != 'control']

        # --- 3. 循环遍历 K 个采样 ---
        for k in range(K):
            pred_expressions = y_samples[k, :, :].cpu().numpy()
            all_pred_X = np.vstack([pred_expressions, basal_expressions])
            adata_pred = ad.AnnData(X=all_pred_X, obs=obs_df, var=var_df)

            # 只对当前的预测样本进行DE分析
            sc.tl.rank_genes_groups(adata_pred, groupby='perturbation', reference='control', method='wilcoxon', key_added="rank_genes_pred")

            des_scores_for_k = {}
            for pert in unique_perts:
                # 提取预先计算好的真实DE结果
                real_de_results = sc.get.rank_genes_groups_df(adata_real, group=pert, key="rank_genes_real")
                # 提取当前样本的预测DE结果
                pred_de_results = sc.get.rank_genes_groups_df(adata_pred, group=pert, key="rank_genes_pred")
                
                # --- DES 计算逻辑 (与之前一致) ---
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