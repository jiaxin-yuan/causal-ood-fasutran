"""
领域泛化的基础模型类
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple


class BaseModel(nn.Module, ABC):
    """
    所有领域泛化模型的抽象基类
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        模型的前向传播
        
        Args:
            x: 输入张量
            **kwargs: 额外参数
            
        Returns:
            输出张量
        """
        pass
    
    @abstractmethod
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        将输入编码为共享特征表示
        
        Args:
            x: 输入张量
            **kwargs: 额外参数
            
        Returns:
            编码后的特征
        """
        pass
    
    def get_shared_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        获取用于领域泛化的共享特征
        
        Args:
            x: 输入张量
            **kwargs: 额外参数
            
        Returns:
            共享特征
        """
        return self.encode(x, **kwargs)
    
    def adapt_to_domain(self, features: torch.Tensor, domain_info: Optional[Dict] = None) -> torch.Tensor:
        """
        将特征适应到特定领域（子类可以重写此方法）
        
        Args:
            features: 输入特征
            domain_info: 领域信息
            
        Returns:
            适应后的特征
        """
        return features
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息用于日志记录
        
        Returns:
            模型信息字典
        """
        return {
            'model_type': self.__class__.__name__,
            'config': self.config,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def save_model(self, path: str):
        """
        保存模型到文件
        
        Args:
            path: 保存模型的路径
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_info': self.get_model_info()
        }, path)
    
    def load_model(self, path: str):
        """
        从文件加载模型
        
        Args:
            path: 加载模型的路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('config', {}), checkpoint.get('model_info', {})


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization layer
    
    AdaIN(z1, z2) = mean(z2) + std(z2) * (z1 - mean(z1)) / std(z1)
    
    Args:
        z1: content features (original input encoded)
        z2: style features (random domain input encoded)
    """
    
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, z1, z2):
        """
        Apply AdaIN normalization
        
        Args:
            z1: content features [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
            z2: style features [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
            
        Returns:
            normalized features with same shape as z1
        """
        # 检查输入是否有NaN
        if torch.isnan(z1).any():
            print("警告：AdaIN z1输入包含NaN值")
            z1 = torch.nan_to_num(z1, nan=0.0)
            
        if torch.isnan(z2).any():
            print("警告：AdaIN z2输入包含NaN值")
            z2 = torch.nan_to_num(z2, nan=0.0)
        
        # 计算统计量
        if z1.dim() == 3:  # [batch_size, seq_len, feature_dim]
            # 在feature_dim维度上计算均值和标准差
            z1_mean = z1.mean(dim=-1, keepdim=True)
            z1_std = z1.std(dim=-1, keepdim=True) + self.eps
            z2_mean = z2.mean(dim=-1, keepdim=True)
            z2_std = z2.std(dim=-1, keepdim=True) + self.eps
        else:  # [batch_size, feature_dim]
            # 在feature_dim维度上计算均值和标准差
            z1_mean = z1.mean(dim=-1, keepdim=True)
            z1_std = z1.std(dim=-1, keepdim=True) + self.eps
            z2_mean = z2.mean(dim=-1, keepdim=True)
            z2_std = z2.std(dim=-1, keepdim=True) + self.eps
        
        # 应用AdaIN公式
        normalized = z2_mean + z2_std * (z1 - z1_mean) / z1_std
        
        # 检查输出是否有NaN
        if torch.isnan(normalized).any():
            print("警告：AdaIN输出包含NaN值")
            normalized = torch.nan_to_num(normalized, nan=0.0)
        
        return normalized 