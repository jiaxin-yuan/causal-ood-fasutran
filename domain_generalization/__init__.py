"""
领域泛化框架
一个用于在域外数据上测试不同模型架构的统一框架。
"""

__version__ = "1.0.0"
__author__ = "领域泛化团队"

from .models import *
from .data_loader import *
from .trainer import *
from .evaluator import *
from .pipeline import DomainGeneralizationPipeline
from .utils import *

__all__ = [
    "TransformerModel", 
    "GraphTransformer",
    "LSTMModel",
    "DataLoader",
    "Trainer",
    "Evaluator",
    "DomainGeneralizationPipeline",
    "create_experiment_config",
    "set_random_seed",
    "print_experiment_summary"
] 