import torch
import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings

# Force matplotlib to use only English and avoid Chinese characters
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# Suppress all matplotlib font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
warnings.filterwarnings('ignore', message='.*font.*')
warnings.filterwarnings('ignore', message='.*findfont.*')
warnings.filterwarnings('ignore', message='.*Font family.*not found.*')
warnings.filterwarnings('ignore', message='.*Arial.*')

from .models import TransformerModel, GraphTransformer, LSTMModel
from .data_loader import DataLoader
from .trainer import Trainer
from .evaluator import Evaluator


class DomainGeneralizationPipeline:
    """领域泛化实验的主要流水线"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化组件
        self.data_loader = DataLoader(config)
        self.models = {}
        self.trainers = {}
        self.evaluators = {}
        self.results = {}
        
        # 创建输出目录
        self.output_dir = config.get('output_dir', 'results')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_model(self, model_type: str, model_config: Dict):
        """创建指定类型的模型"""
        
        if model_type == 'transformer':
            model = TransformerModel(model_config)
        elif model_type == 'graph_transformer':
            model = GraphTransformer(model_config)
        elif model_type == 'lstm':
            model = LSTMModel(model_config)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        return model
    
    def setup_experiment(self, model_configs: Dict[str, Dict]):
        """为实验设置模型"""
        
        # 创建日志目录
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        for model_name, config in model_configs.items():
            print(f"正在设置 {model_name}...")
            
            # 创建模型
            model = self.create_model(config['type'], config)
            self.models[model_name] = model
            
            # 创建训练器
            trainer = Trainer(model, config)
            
            # 设置日志系统
            import logging
            from datetime import datetime
            
            # 创建日志文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{model_name}_{timestamp}.log"
            log_filepath = os.path.join(log_dir, log_filename)
            
            # 创建logger
            logger = logging.getLogger(f"trainer_{model_name}")
            logger.setLevel(logging.INFO)
            
            # 清除之前的handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # 创建文件handler
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 创建控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.propagate = False
            
            # 将logger添加到trainer
            trainer.logger = logger
            trainer.model_name = model_name
            
            logger.info(f"为模型 {model_name} 创建了日志系统")
            logger.info(f"日志文件: {log_filepath}")
            self.trainers[model_name] = trainer
            
            # 创建评估器
            evaluator = Evaluator(model, config)
            evaluator.set_data_loader(self.data_loader)  # 设置数据加载器以支持反归一化
            self.evaluators[model_name] = evaluator
            
            print(f"✓ {model_name} 设置完成")
    
    def run_experiment(self, data_config: Dict, model_configs: Dict[str, Dict]):
        """运行完整的领域泛化实验"""
        
        print("=" * 60)
        print("领域泛化实验")
        print("=" * 60)
        
        # 设置模型
        self.setup_experiment(model_configs)
        
        # 生成或加载数据
        if data_config.get('use_synthetic', True):
            print("\n正在生成合成数据...")
            data, labels, domains = self.data_loader.generate_synthetic_data(
                num_samples=data_config.get('num_samples', 1000),
                seq_len=data_config.get('seq_len', 50),
                input_dim=data_config.get('input_dim', 128),
                num_domains=data_config.get('num_domains', 3),
                num_classes=data_config.get('num_classes', 5)
            )
        else:
            print(f"\n正在从 {data_config['data_path']} 加载数据...")
            
            # 获取域信息
            train_domains = data_config.get('train_domains', [0])
            val_domains = data_config.get('val_domains', [1])
            test_domains = data_config.get('test_domains', [2])
            
            # 使用新的基于域的归一化方法
            print("使用基于训练域的归一化方法...")
            data, labels, domains = self.data_loader.load_real_data_with_domain_normalization(
                data_config['data_path'], train_domains, val_domains, test_domains
            )
        
        # 按领域分割数据
        train_domains = data_config.get('train_domains', [0])
        val_domains = data_config.get('val_domains', [1])
        test_domains = data_config.get('test_domains', [2])
        
        train_data, val_data, test_data = self.data_loader.split_data_by_domain(
            data, labels, domains, train_domains, val_domains, test_domains,
            random_split=data_config.get('random_split', False)
        )
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.data_loader.create_dataloaders(
            train_data, val_data, test_data
        )
        
        print(f"数据分割: 训练={len(train_data[0])}, 验证={len(val_data[0])}, 测试={len(test_data[0])}")
        if len(test_data[0]) == 0:
            print(f"警告：test_domains={test_domains} 没有分到任何测试数据，请检查 domain 分布！")
        
        # 如果使用了基于域的归一化，打印归一化参数
        if hasattr(self.data_loader, 'normalization_params') and self.data_loader.normalization_params:
            print(f"\n使用的归一化参数:")
            params = self.data_loader.normalization_params
            print(f"  方法: {params['method']}")
            print(f"  训练域 rtime 统计:")
            print(f"    最小值: {params['rtime_min_train']:.4f}")
            print(f"    最大值: {params['rtime_max_train']:.4f}")
            print(f"    均值: {params['rtime_mean_train']:.4f}")
            print(f"    标准差: {params['rtime_std_train']:.4f}")
        
        # 训练和评估每个模型
        for model_name in self.models.keys():
            print(f"\n{'='*20} {model_name.upper()} {'='*20}")
            
            # 训练模型
            model_save_path = os.path.join(self.output_dir, f"{model_name}_best.pth")
            train_history = self.trainers[model_name].train(
                train_loader, val_loader, model_save_path
            )
            
            # 评估模型
            results = self.evaluators[model_name].evaluate(test_loader)
            self.results[model_name] = results
            
            # 生成图表
            self._generate_model_plots(model_name, results, train_history)
            
            print(f"✓ {model_name} 实验完成")
        
        # 比较所有模型
        self._compare_all_models()
        
        # 保存结果
        self._save_results()
        
        print(f"\n{'='*60}")
        print("实验完成")
        print(f"结果已保存到: {self.output_dir}")
        print("=" * 60)
    
    def _generate_model_plots(self, model_name: str, results: Dict, train_history: Dict):
        """为特定模型生成图表"""
        
        # 创建模型特定目录
        model_dir = os.path.join(self.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 训练曲线 - 只显示损失，因为这是回归任务
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_history['train_losses'], label='Training Loss', color='blue')
        plt.plot(train_history['val_losses'], label='Validation Loss', color='red')
        plt.title(f'{model_name} - Training Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # 计算MAE历史（如果有的话）
        if 'train_mae' in train_history:
            plt.plot(train_history['train_mae'], label='Training MAE', color='green')
            plt.plot(train_history['val_mae'], label='Validation MAE', color='orange')
            plt.title(f'{model_name} - MAE Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Absolute Error')
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # 如果没有MAE历史，显示学习率变化
            plt.plot(range(len(train_history['train_losses'])), 
                    [train_history.get('learning_rates', [1e-6] * len(train_history['train_losses']))[i] 
                     for i in range(len(train_history['train_losses']))], 
                    label='Learning Rate', color='purple')
            plt.title(f'{model_name} - Learning Rate Changes')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 预测 vs 真实值散点图 (回归任务专用)
        if 'predictions' in results and 'labels' in results:
            plt.figure(figsize=(8, 8))
            predictions = results['predictions']
            labels = results['labels']
            
            plt.scatter(labels, predictions, alpha=0.6, s=20)
            
            # 绘制理想线 (y=x)
            min_val = min(min(labels), min(predictions))
            max_val = max(max(labels), max(predictions))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{model_name} - Prediction vs True Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 添加R²和MAE信息
            from sklearn.metrics import r2_score, mean_absolute_error
            r2 = r2_score(labels, predictions)
            mae = mean_absolute_error(labels, predictions)
            plt.text(0.05, 0.95, f'R² = {r2:.4f}\nMAE = {mae:.4f}', 
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 残差图
        if 'predictions' in results and 'labels' in results:
            plt.figure(figsize=(10, 6))
            predictions = results['predictions']
            labels = results['labels']
            residuals = predictions - labels
            
            plt.subplot(1, 2, 1)
            plt.scatter(predictions, residuals, alpha=0.6, s=20)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals (Predicted - True)')
            plt.title(f'{model_name} - Residual Plot')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title(f'{model_name} - Residual Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(model_dir, 'residual_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 生成报告
        if hasattr(self.evaluators[model_name], 'generate_report'):
            self.evaluators[model_name].generate_report(
                results,
                os.path.join(model_dir, 'evaluation_report.txt')
            )
    
    def _compare_all_models(self):
        """比较所有模型"""
        
        comparison_df = self.evaluators[list(self.evaluators.keys())[0]].compare_models(
            self.results,
            os.path.join(self.output_dir, 'model_comparison.png')
        )
        
        # 保存比较结果
        comparison_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), index=False)
        
        return comparison_df
    
    def _save_results(self):
        """保存所有结果"""
        
        # 保存结果摘要
        summary = {}
        for model_name, results in self.results.items():
            summary[model_name] = {
                'metrics': results['metrics'],
                'model_info': self.models[model_name].get_model_info()
            }
        
        with open(os.path.join(self.output_dir, 'results_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # 保存详细结果
        detailed_results = {}
        for model_name, results in self.results.items():
            detailed_results[model_name] = {
                'predictions': results['predictions'].tolist(),
                'labels': results['labels'].tolist(),
                'domains': results['domains'].tolist(),
                'features': results['features'].tolist(),
                'metrics': results['metrics']
            }
        
        with open(os.path.join(self.output_dir, 'detailed_results.json'), 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
    
    def load_experiment(self, results_path: str):
        """从之前的实验加载结果"""
        
        with open(os.path.join(results_path, 'results_summary.json'), 'r') as f:
            summary = json.load(f)
        
        with open(os.path.join(results_path, 'detailed_results.json'), 'r') as f:
            detailed_results = json.load(f)
        
        # 转换回numpy数组
        for model_name, results in detailed_results.items():
            results['predictions'] = np.array(results['predictions'])
            results['labels'] = np.array(results['labels'])
            results['domains'] = np.array(results['domains'])
            results['features'] = np.array(results['features'])
        
        self.results = detailed_results
        return summary, detailed_results
    
    def get_best_model(self, metric: str = 'mae') -> Tuple[str, float]:
        """根据指定指标获取性能最佳的模型"""
        
        best_model = None
        best_score = float('inf')  # 对于MAE，越小越好
        
        for model_name, results in self.results.items():
            metrics = results['metrics']
            
            # 构建完整的指标名称
            if metric == 'mae':
                full_metric_name = 'overall_mae'
            elif metric == 'mse':
                full_metric_name = 'overall_mse'
            elif metric == 'rmse':
                full_metric_name = 'overall_rmse'
            elif metric == 'r2':
                full_metric_name = 'overall_r2'
            else:
                full_metric_name = metric
            
            if full_metric_name in metrics:
                score = metrics[full_metric_name]
                # 对于MAE、MSE和RMSE，越小越好
                if metric in ['mae', 'mse', 'rmse']:
                    if score < best_score:
                        best_score = score
                        best_model = model_name
                # 对于R²，越大越好
                elif metric in ['r2', 'r2_score']:
                    if score > best_score:
                        best_score = score
                        best_model = model_name
        
        return best_model, best_score 