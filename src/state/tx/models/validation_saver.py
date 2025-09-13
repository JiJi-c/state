from dataclasses import dataclass
from typing import List, Dict, Any
import torch
import pickle
from pathlib import Path

@dataclass
class ValidationBatch:
    """单个validation batch的数据"""
    predictions: torch.Tensor      # [B, L, G] 预测表达值
    targets: torch.Tensor         # [B, L, G] 真实表达值  
    basals: torch.Tensor          # [B, L, G] 对照基因表达值
    pert_names: List[str]         # [B*L] 扰动名称
    
class ValidationSaver:
    """Validation数据收集和保存器"""
    
    def __init__(self):
        self.batches: List[ValidationBatch] = []
        
    def add_batch(
        self, 
        predictions: torch.Tensor,
        targets: torch.Tensor, 
        basals: torch.Tensor,
        pert_names: List[str],
    ):
        """添加一个validation batch"""
        batch = ValidationBatch(
            predictions=predictions,
            targets=targets,
            basals=basals, 
            pert_names=pert_names,
        )
        self.batches.append(batch)    

    def get_batches(self):
        """获取所有batch数据"""
        predictions = torch.cat([b.predictions for b in self.batches], dim=1)
        targets = torch.cat([b.targets for b in self.batches], dim=0)
        basals = torch.cat([b.basals for b in self.batches], dim=0)
        pert_names = [name for b in self.batches for name in b.pert_names]

        return predictions, targets, basals, pert_names

    def clear(self):
        """清空缓存的batch数据"""
        self.batches.clear()
        
