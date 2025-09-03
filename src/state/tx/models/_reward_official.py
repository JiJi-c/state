import torch
import numpy as np
import pandas as pd
import polars as pl
import anndata as ad
from typing import Dict, List, Tuple, Optional
import logging
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

from cell_eval._types import PerturbationAnndataPair, initialize_de_comparison, DEComparison
from cell_eval.metrics import discrimination_score, de_overlap_metric, mae
from pdex import parallel_differential_expression

class MetricsEvaluator:
    def __init__(
        self,
        gene_names: List[str],
        reward_weights: Dict[str, float],
        control_pert: str = "non-targeting",
        pert_col: str = "target_gene",
        de_method: str = "wilcoxon",
        num_threads: int = 48,
        device: str = "cuda"
    ):
        """
            初始化奖励计算器。
            
            Args:
                gene_names: 基因名称列表 (长度为G)。
                reward_weights: 各指标的权重, e.g., {"pds": 0.45, "des": 0.45}。
                control_pert: 控制组扰动的名称。
                pert_col: AnnData中扰动列的名称。
                de_method: 差分表达分析方法。
                num_threads: 用于DE计算的并行线程数。
                n_cells_per_pert: 为每个扰动模拟的细胞数，用于统计检验。
                device: PyTorch设备。
        """
        self.gene_names = gene_names
        self.n_genes = len(gene_names)
        self.reward_weights = reward_weights
        self.control_pert = control_pert
        self.pert_col = pert_col
        self.de_method = de_method
        self.num_threads = num_threads
        self.device = device

        pl.enable_string_cache()

    def compute_rewards(
        self,
        y_samples: torch.Tensor,      # [K, N, G] 
        y_real: torch.Tensor,         # [N, G]
        basal_cells: torch.Tensor,    # [N, G]
        pert_indices: torch.Tensor,   # [N]
        all_pert_names: List[str],    # 扰动名称总列表
    ) -> Tuple[torch.Tensor, ...]:
        """
        计算整个批次的奖励。
        
        Returns:
            final_rewards: [K, N] - 每个采样、每个细胞的最终奖励。
            all_components: Dict - 各指标的奖励分数 [K, M_local]。
            M_local: int - 当前batch中实际出现的扰动数量。
        """
        K, N, G = y_samples.shape
        
        # 1. 确定当前batch中实际出现了哪些扰动
        local_perts, local_indices = pert_indices.unique(sorted=True, return_inverse=True)
        M_local = len(local_perts) # 0,1,3,9    # 1 0 3 2 1 2 
        local_pert_names = [all_pert_names[i] for i in local_perts.cpu().numpy()]

        # 2. 在扰动级别上计算每个采样的奖励
        all_pert_rewards = []
        all_components = {}

        for k in range(K):
            # 对每个采样k，计算其在M_local个扰动上的奖励
            pert_rewards_k, components_k = self._compute_single_sample_reward(
                y_samples[k],       # [N, G]
                y_real,             # [N, G]
                basal_cells,        # [N, G]
                local_indices,  # [N] -> 扰动在local list中的索引
                local_pert_names,   # 当前batch出现的扰动名列表
                M_local
            )
            all_pert_rewards.append(pert_rewards_k)

            for key, val in components_k.items():
                if key not in all_components:
                    all_components[key] = []
                all_components[key].append(val)
        
        # 3. 组合并映射回细胞级别
        # all_pert_rewards: List of [M_local] -> [K, M_local]
        rewards_tensor = torch.stack(all_pert_rewards)
        # 使用local_indices_map将扰动奖励映射回每个细胞
        final_rewards = rewards_tensor[:, local_indices]  # [K, N]
    
        for key in all_components:
            all_components[key] = torch.stack(all_components[key]) # [K, M_local]

        return final_rewards, all_components, M_local
        
    def _compute_single_sample_reward(
            self,
            y_sample: torch.Tensor,       # [N, G]
            y_real: torch.Tensor,         # [N, G]  
            basal_cells: torch.Tensor,    # [N, G]
            local_indices: torch.Tensor, # [N]
            local_pert_names: List[str],   # List[str]
            M_local: int
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """为单个采样k计算奖励，返回一个[M_local]的tensor。"""

        pred_adata, real_adata = self._create_anndata_objects(
            y_sample, y_real, basal_cells, local_indices, local_pert_names, M_local
        )

        anndata_pair = PerturbationAnndataPair(
                real=real_adata, pred=pred_adata,
                pert_col=self.pert_col, control_pert=self.control_pert
        )

        de_comparison = self._compute_de_comparison(anndata_pair)

        # 3. 计算各项指标
        components = {}
        
        # PDS 计算
        pds_scores_dict = discrimination_score(anndata_pair, metric="l1", exclude_target_gene=True)
        components["pds"] = self._dict_to_tensor(pds_scores_dict, local_pert_names, M_local)
        
        # DES 计算
        des_scores_dict = de_overlap_metric(de_comparison, k=None, metric="overlap")
        components["des"] = self._dict_to_tensor(des_scores_dict, local_pert_names, M_local)
        
        # 4. 加权组合奖励
        total_reward = torch.zeros(M_local, device=self.device)
        total_weight = sum(self.reward_weights.values())
        
        for metric, weight in self.reward_weights.items():
            reward_val = components[metric]
            reward_val = torch.where(torch.isnan(reward_val), 0.0, reward_val)
            total_reward += weight * reward_val

        total_reward /= total_weight
            
        return total_reward, components


    def _create_anndata_objects(
        self, 
        y_sample_k: torch.Tensor,       # [N, G], N=131072
        y_real: torch.Tensor,           # [N, G]
        basal_cells: torch.Tensor,      # [N, G]
        local_indices: torch.Tensor,    # [N], e.g., [0, 1, 0, 2, ...]
        local_pert_names: List[str],    # e.g., ["AHR", "BCL2", ...]
        M_local: int
    ) -> Tuple[ad.AnnData, ad.AnnData]:
        """
        【新版本】从真实的细胞级Tensors创建AnnData对象。
        这个版本不再使用np.repeat，而是直接利用batch中的细胞。
        """
        
        # 将Tensors转为Numpy
        y_sample_np = y_sample_k.detach().cpu().numpy()
        y_real_np = y_real.detach().cpu().numpy()
        
        # --- 1. 找到控制组细胞 ---
        control_cells_expression = basal_cells.detach().cpu().numpy()

        # --- 2. 准备AnnData的 `.obs` (细胞元数据) ---
        # `local_indices` 已经是每个细胞对应的扰动索引了。
        # 我们需要把它转换成扰动名称。
        pert_names_per_cell = [local_pert_names[i] for i in local_indices.cpu().numpy()]
        
        # 创建预测数据的obs
        pred_obs_df = pd.DataFrame({
            self.pert_col: pert_names_per_cell
        })
        # 创建真实数据的obs
        real_obs_df = pd.DataFrame({
            self.pert_col: pert_names_per_cell
        })

        # 3. 组合成最终的表达矩阵
        # 将扰动细胞和控制细胞堆叠起来
        final_pred_X = np.vstack([y_sample_np, control_cells_expression])
        final_real_X = np.vstack([y_real_np, control_cells_expression])
        
        # 4. 组合最终的 .obs
        control_labels = [self.control_pert] * control_cells_expression.shape[0]
        
        # 将扰动细胞的标签和控制细胞的标签合并
        final_pred_obs = pd.concat([
            pred_obs_df,
            pd.DataFrame({self.pert_col: control_labels})
        ], ignore_index=True)

        final_real_obs = pd.concat([
            real_obs_df,
            pd.DataFrame({self.pert_col: control_labels})
        ], ignore_index=True)

        # 5. 创建 `.var` (基因元数据)
        var_df = pd.DataFrame(index=self.gene_names)
        
        # 6. 创建AnnData对象
        pred_adata = ad.AnnData(X=final_pred_X, obs=final_pred_obs, var=var_df)
        real_adata = ad.AnnData(X=final_real_X, obs=final_real_obs, var=var_df)
        
        return pred_adata, real_adata

    def _compute_de_comparison(self, anndata_pair: PerturbationAnndataPair) -> DEComparison:
        """封装DE计算过程。"""
        common_kwargs = {
            "reference": self.control_pert,
            "groupby_key": self.pert_col,
            "num_workers": self.num_threads,
            "metric": self.de_method,
            "as_polars": True
        }
        real_de = parallel_differential_expression(adata=anndata_pair.real, **common_kwargs)
        pred_de = parallel_differential_expression(adata=anndata_pair.pred, **common_kwargs)
        
        return initialize_de_comparison(real=real_de, pred=pred_de)

    def _dict_to_tensor(self, score_dict: Dict[str, float], ordered_pert_names: List[str], M_local: int) -> torch.Tensor:
        """将返回的字典按顺序转换为Tensor。"""
        scores = [score_dict.get(name, 0.0) for name in ordered_pert_names]
        return torch.tensor(scores, device=self.device, dtype=torch.float32)