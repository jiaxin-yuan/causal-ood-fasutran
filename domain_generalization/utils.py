import torch
import numpy as np
import json
import os
import random
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns


def set_random_seed(seed: int = 42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_experiment_config(
    model_types: List[str] = None,
    data_config: Dict = None,
    training_config: Dict = None,
    adain_flag: bool=False
) -> Dict:
    """创建完整的实验配置"""
    
    if model_types is None:
        model_types = ['transformer', 'graph_transformer', 'lstm']
    
    if data_config is None:
        data_config = {
            'use_synthetic': True,
            'num_samples': 1000,
            'seq_len': 50,
            'input_dim': 128,
            'num_domains': 3,
            'num_classes': 5,
            'train_domains': [0],
            'val_domains': [1],
            'test_domains': [2]
        }
    
    if training_config is None:
        training_config = {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'num_epochs': 50,
            'early_stopping_patience': 10,
            'seed': 42
        }
    
    # 模型配置
    model_configs = {}
    
    # 确定输出维度 - 对于回归任务是1，对于分类任务是类别数
    output_dim = data_config.get('num_classes', 1)  # 默认为1（回归任务）

    
    for model_type in model_types:
        if model_type == 'transformer':
            model_configs['transformer'] = {
                'type': 'transformer',
                'input_dim': data_config['input_dim'],
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'output_dim': output_dim,
                'dropout': 0.1,
                'use_adain': adain_flag,
                'adain_prob': 0.8,
                **training_config
            }
        elif model_type == 'graph_transformer':
            model_configs['graph_transformer'] = {
                'type': 'graph_transformer',
                'input_dim': data_config['input_dim'],
                'hidden_dim': 256,
                'output_dim': output_dim,
                'num_heads': 8,
                'num_layers': 2,
                'dropout': 0.1,
                'use_adain': adain_flag,
                'adain_prob': 0.8,
                **training_config
            }
        elif model_type == 'lstm':
            model_configs['lstm'] = {
                'type': 'lstm',
                'input_dim': data_config['input_dim'],
                'hidden_dim': 256,
                'output_dim': output_dim,
                'num_layers': 2,
                'dropout': 0.1,
                'bidirectional': True,
                'use_adain': adain_flag,
                'adain_prob': 0.8,
                **training_config
            }
    
    return {
        'data_config': data_config,
        'model_configs': model_configs,
        'output_dir': 'results',
        'seed': training_config['seed']
    }


def plot_training_history(train_history: Dict, save_path: str = None):
    """绘制训练历史"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失图
    axes[0].plot(train_history['train_losses'], label='训练损失', color='blue')
    axes[0].plot(train_history['val_losses'], label='验证损失', color='red')
    axes[0].set_title('训练和验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率图
    axes[1].plot(train_history['train_accuracies'], label='训练准确率', color='blue')
    axes[1].plot(train_history['val_accuracies'], label='验证准确率', color='red')
    axes[1].set_title('训练和验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率 (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_domain_shift(features: np.ndarray, domains: np.ndarray, save_path: str = None):
    """分析学习特征中的领域偏移"""
    
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # PCA分析
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    # t-SNE分析
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    unique_domains = np.unique(domains)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_domains)))
    
    # PCA图
    for i, domain in enumerate(unique_domains):
        domain_mask = domains == domain
        axes[0].scatter(
            features_pca[domain_mask, 0], 
            features_pca[domain_mask, 1],
            c=[colors[i]], 
            label=f'领域 {domain}',
            alpha=0.7
        )
    axes[0].set_title('特征的PCA可视化')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()
    
    # t-SNE图
    for i, domain in enumerate(unique_domains):
        domain_mask = domains == domain
        axes[1].scatter(
            features_tsne[domain_mask, 0], 
            features_tsne[domain_mask, 1],
            c=[colors[i]], 
            label=f'领域 {domain}',
            alpha=0.7
        )
    axes[1].set_title('特征的t-SNE可视化')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_domain_statistics(features: np.ndarray, domains: np.ndarray) -> Dict:
    """计算领域特定的统计信息"""
    
    unique_domains = np.unique(domains)
    domain_stats = {}
    
    for domain in unique_domains:
        domain_mask = domains == domain
        domain_features = features[domain_mask]
        
        domain_stats[f'domain_{domain}'] = {
            'mean': np.mean(domain_features, axis=0),
            'std': np.std(domain_features, axis=0),
            'samples': len(domain_features),
            'feature_dim': domain_features.shape[1]
        }
    
    # 计算领域距离
    domain_distances = {}
    for i, domain1 in enumerate(unique_domains):
        for j, domain2 in enumerate(unique_domains):
            if i < j:
                mean1 = domain_stats[f'domain_{domain1}']['mean']
                mean2 = domain_stats[f'domain_{domain2}']['mean']
                
                # 欧几里得距离
                euclidean_dist = np.linalg.norm(mean1 - mean2)
                
                # 余弦距离
                cosine_dist = 1 - np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2))
                
                domain_distances[f'{domain1}_to_{domain2}'] = {
                    'euclidean': euclidean_dist,
                    'cosine': cosine_dist
                }
    
    return {
        'domain_stats': domain_stats,
        'domain_distances': domain_distances
    }


def save_experiment_config(config: Dict, save_path: str):
    """将实验配置保存到文件"""
    
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)


def load_experiment_config(config_path: str) -> Dict:
    """从文件加载实验配置"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def print_experiment_summary(results: Dict):
    """打印实验结果摘要"""
    
    print("\n" + "="*60)
    print("实验摘要")
    print("="*60)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name.upper()}:")
        
        # 检查是否为回归任务
        if 'overall_mse' in metrics:
            # 回归任务指标
            print(f"  整体MAE: {metrics.get('overall_mae', 'N/A'):.4f}")
            print(f"  整体RMSE: {metrics.get('overall_rmse', 'N/A'):.4f}")
            print(f"  整体R²: {metrics.get('overall_r2', 'N/A'):.4f}")
            print(f"  领域差距: {metrics.get('domain_gap', 'N/A'):.4f}")
            print(f"  领域数量: {metrics.get('num_domains', 'N/A')}")
            
            # 原始尺度指标（如果有的话）
            if 'overall_mae_original' in metrics:
                print(f"  原始尺度MAE: {metrics['overall_mae_original']:.4f}")
                print(f"  原始尺度RMSE: {metrics.get('overall_rmse_original', 'N/A'):.4f}")
            
            print("  每个领域的性能:")
            for domain, domain_metric in metrics.get('domain_metrics', {}).items():
                mae = domain_metric.get('mae', 'N/A')
                rmse = domain_metric.get('rmse', 'N/A')
                samples = domain_metric.get('samples', 'N/A')
                print(f"    {domain}: MAE={mae:.4f}, RMSE={rmse:.4f}, 样本数={samples}")
        
        elif 'overall_accuracy' in metrics:
            # 分类任务指标（保留原有逻辑）
            print(f"  整体准确率: {metrics['overall_accuracy']:.4f}")
            print(f"  整体F1分数: {metrics['overall_f1']:.4f}")
            print(f"  领域差距: {metrics['domain_gap']:.4f}")
            print(f"  领域数量: {metrics['num_domains']}")
            
            print("  每个领域的性能:")
            for domain, domain_metric in metrics['domain_metrics'].items():
                print(f"    {domain}: 准确率={domain_metric['accuracy']:.4f}, F1={domain_metric['f1']:.4f}")
        
        else:
            print("  ❌ 未找到有效的评估指标")
    
    # 找到最佳模型
    best_model = None
    best_score = float('inf')  # 对于回归任务，越小越好
    
    for model_name, result in results.items():
        metrics = result['metrics']
        
        # 优先使用MAE作为评估指标
        if 'overall_mae' in metrics:
            score = metrics['overall_mae']
        elif 'overall_accuracy' in metrics:
            score = -metrics['overall_accuracy']  # 分类任务，准确率越高越好，所以用负值
        else:
            continue
            
        if score < best_score:
            best_score = score
            best_model = model_name
    
    if best_model:
        if 'overall_mae' in results[best_model]['metrics']:
            print(f"\n🎉 最佳模型: {best_model} (MAE: {best_score:.4f})")
        else:
            print(f"\n🎉 最佳模型: {best_model} (准确率: {-best_score:.4f})")
    else:
        print(f"\n❌ 无法确定最佳模型")
    
    print("="*60) 