import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
import pandas as pd
from typing import Dict, List, Tuple
import os
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

class Evaluator:
    """用于领域泛化模型的评估器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.data_loader = None  # 将在pipeline中设置
        
    def set_data_loader(self, data_loader):
        """设置数据加载器以支持反归一化"""
        self.data_loader = data_loader
        
    def evaluate(self, test_loader, domain_info=None):
        """在测试数据上评估模型"""
        self.model.eval()
        
        print(f"Test loader batch count: {len(test_loader)}")
        
        # 检查测试集是否为空
        if len(test_loader) == 0:
            print("Warning: Test loader is empty! No data to evaluate.")
            return {
                'predictions': np.array([]),
                'labels': np.array([]),
                'domains': np.array([]),
                'features': np.array([]),
                'metrics': {}
            }
        
        all_predictions = []
        all_labels = []
        all_domains = []
        all_features = []
        
        batch_count = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch_count += 1
                data = batch['data'].float().to(self.device)
                labels = batch['rtime'].float().to(self.device)
                domains = batch['domain'].long().to(self.device)
                
                batch_size = data.size(0)
                total_samples += batch_size
                
                # 只在特定间隔打印进度信息
                if batch_count % 100 == 0 or batch_count == 1:
                    print(f"处理第 {batch_count} 个批次，包含 {batch_size} 个样本 (总计: {total_samples})")
                
                try:
                    # 前向传播
                    outputs = self.model(data).squeeze(-1)  # 回归任务，移除最后一个维度
                    features = self.model.get_shared_features(data)
                    
                    # 存储结果 (回归任务直接使用模型输出)
                    all_predictions.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_domains.extend(domains.cpu().numpy())
                    all_features.append(features.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in batch {batch_count}: {str(e)}")
                    continue
        
        print(f"总共处理了 {total_samples} 个测试样本")
        
        # 转换为numpy数组
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_domains = np.array(all_domains)
        
        if len(all_features) == 0:
            print("Warning: No features collected in evaluation. Test set may be empty or model failed.")
            all_features = np.array([])
            return {
                'predictions': all_predictions,
                'labels': all_labels,
                'domains': all_domains,
                'features': all_features,
                'metrics': {}
            }
        else:
            all_features = np.concatenate(all_features, axis=0)
            print(f"成功收集到 {len(all_predictions)} 个预测结果和 {all_features.shape[0]} 个特征向量")
        
        # 计算指标
        metrics = self._calculate_metrics(all_predictions, all_labels, all_domains)
        
        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'domains': all_domains,
            'features': all_features,
            'metrics': metrics
        }
    
    def _calculate_metrics(self, predictions, labels, domains):
        """计算回归评估指标"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # 计算归一化尺度下的指标
        overall_mse = mean_squared_error(labels, predictions)
        overall_mae = mean_absolute_error(labels, predictions)
        overall_rmse = np.sqrt(overall_mse)
        overall_r2 = r2_score(labels, predictions)
        
        # 如果有数据加载器，计算原始尺度下的指标
        original_scale_metrics = {}
        if self.data_loader and hasattr(self.data_loader, 'inverse_normalize_rtime'):
            try:
                # 转换回原始尺度
                original_labels = self.data_loader.inverse_normalize_rtime(labels)
                original_predictions = self.data_loader.inverse_normalize_rtime(predictions)
                
                # 计算原始尺度的指标
                original_mse = mean_squared_error(original_labels, original_predictions)
                original_mae = mean_absolute_error(original_labels, original_predictions)
                original_rmse = np.sqrt(original_mse)
                original_r2 = r2_score(original_labels, original_predictions)
                
                original_scale_metrics = {
                    'overall_mse_original': original_mse,
                    'overall_mae_original': original_mae,
                    'overall_rmse_original': original_rmse,
                    'overall_r2_original': original_r2
                }
                
                print(f"原始尺度指标:")
                print(f"  MSE: {original_mse:.4f}")
                print(f"  MAE: {original_mae:.4f}")
                print(f"  RMSE: {original_rmse:.4f}")
                print(f"  R²: {original_r2:.4f}")
                
            except Exception as e:
                print(f"警告：无法计算原始尺度指标: {e}")
        
        # 每个领域的指标
        unique_domains = np.unique(domains)
        domain_metrics = {}
        
        for domain in unique_domains:
            domain_mask = domains == domain
            domain_labels = labels[domain_mask]
            domain_predictions = predictions[domain_mask]
            
            if len(domain_labels) > 0:
                domain_mse = mean_squared_error(domain_labels, domain_predictions)
                domain_mae = mean_absolute_error(domain_labels, domain_predictions)
                domain_rmse = np.sqrt(domain_mse)
                domain_r2 = r2_score(domain_labels, domain_predictions) if len(domain_labels) > 1 else 0.0
                
                domain_metric = {
                    'mse': domain_mse,
                    'mae': domain_mae,
                    'rmse': domain_rmse,
                    'r2': domain_r2,
                    'samples': len(domain_labels)
                }
                
                # 添加原始尺度的域指标
                if self.data_loader and hasattr(self.data_loader, 'inverse_normalize_rtime'):
                    try:
                        orig_domain_labels = self.data_loader.inverse_normalize_rtime(domain_labels)
                        orig_domain_predictions = self.data_loader.inverse_normalize_rtime(domain_predictions)
                        
                        domain_metric.update({
                            'mse_original': mean_squared_error(orig_domain_labels, orig_domain_predictions),
                            'mae_original': mean_absolute_error(orig_domain_labels, orig_domain_predictions),
                            'rmse_original': np.sqrt(mean_squared_error(orig_domain_labels, orig_domain_predictions)),
                            'r2_original': r2_score(orig_domain_labels, orig_domain_predictions) if len(orig_domain_labels) > 1 else 0.0
                        })
                    except Exception as e:
                        print(f"警告：无法计算域 {domain} 的原始尺度指标: {e}")
                
                domain_metrics[f'domain_{domain}'] = domain_metric
        
        # 领域泛化差距 (使用MAE)
        domain_maes = [domain_metrics[f'domain_{d}']['mae'] for d in unique_domains if f'domain_{d}' in domain_metrics]
        domain_gap = max(domain_maes) - min(domain_maes) if domain_maes else 0.0
        
        # 原始尺度的领域差距
        original_domain_gap = 0.0
        if self.data_loader and hasattr(self.data_loader, 'inverse_normalize_rtime'):
            try:
                original_domain_maes = [domain_metrics[f'domain_{d}'].get('mae_original', 0) for d in unique_domains if f'domain_{d}' in domain_metrics and 'mae_original' in domain_metrics[f'domain_{d}']]
                original_domain_gap = max(original_domain_maes) - min(original_domain_maes) if original_domain_maes else 0.0
            except Exception as e:
                print(f"警告：无法计算原始尺度的领域差距: {e}")
        
        # 合并所有指标
        metrics = {
            'overall_mse': overall_mse,
            'overall_mae': overall_mae,
            'overall_rmse': overall_rmse,
            'overall_r2': overall_r2,
            'domain_metrics': domain_metrics,
            'domain_gap': domain_gap,
            'num_domains': len(unique_domains)
        }
        
        # 添加原始尺度指标
        metrics.update(original_scale_metrics)
        if original_domain_gap > 0:
            metrics['domain_gap_original'] = original_domain_gap
        
        return metrics
    
    def plot_regression_scatter(self, predictions, labels, save_path=None):
        """绘制回归散点图"""
        if predictions is None or labels is None or len(predictions) == 0 or len(labels) == 0:
            print("Warning: No predictions or labels to plot regression scatter. Skip regression scatter plot.")
            return
        
        plt.figure(figsize=(10, 8))
        plt.scatter(labels, predictions, alpha=0.6)
        
        # 绘制理想线 (y=x)
        min_val = min(min(labels), min(predictions))
        max_val = max(max(labels), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Line')
        
        plt.title('Regression Prediction Scatter Plot')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_domain_performance(self, metrics, save_path=None):
        """绘制跨领域性能"""
        if not metrics or 'domain_metrics' not in metrics or not metrics['domain_metrics']:
            print("Warning: No domain metrics to plot domain performance. Skip domain performance plot.")
            return
        domain_metrics = metrics['domain_metrics']
        domains = list(domain_metrics.keys())
        maes = [domain_metrics[d]['mae'] for d in domains]
        plt.figure(figsize=(10, 6))
        plt.bar(domains, maes, color='skyblue')
        plt.title('Cross-Domain Performance (MAE)')
        plt.xlabel('Domain')
        plt.ylabel('Mean Absolute Error (MAE)')
        for i, v in enumerate(maes):
            plt.text(i, v + max(maes) * 0.01, f'{v:.3f}', ha='center', va='bottom')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_visualization(self, features, domains, save_path=None):
        """使用t-SNE可视化学习到的特征"""
        if features is None or len(features) == 0 or domains is None or len(domains) == 0:
            print("Warning: No features or domains to plot feature visualization. Skip feature visualization plot.")
            return
        n_samples = features.shape[0]
        perplexity = min(30, n_samples - 1)
        if perplexity < 1:
            print("样本数太少，无法进行t-SNE可视化。跳过特征可视化。")
            return
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        features_2d = tsne.fit_transform(features)
        plt.figure(figsize=(10, 8))
        unique_domains = np.unique(domains)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_domains)))
        for i, domain in enumerate(unique_domains):
            domain_mask = domains == domain
            plt.scatter(
                features_2d[domain_mask, 0], 
                features_2d[domain_mask, 1],
                c=[colors[i]], 
                label=f'Domain {domain}',
                alpha=0.7
            )
        plt.title('t-SNE Visualization of Learned Features')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, results, save_path=None):
        """生成综合评估报告"""
        metrics = results['metrics']
        
        # 检查是否有任何数据
        if not results['predictions'].size or not results['labels'].size:
            error_msg = "测试集为空，无法生成评估报告。请检查数据配置和领域设置。"
            print(f"Warning: {error_msg}")
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(f"评估报告\n{'='*50}\n错误: {error_msg}\n")
            return None
        
        # 检查metrics有效性，但允许部分缺失
        if not metrics:
            print("Warning: 评估指标为空，生成基础报告。")
            metrics = {'error': '指标计算失败'}
        report = f"""
领域泛化评估报告
=====================================

测试样本数: {len(results['predictions'])}
测试领域: {np.unique(results['domains']) if results['domains'].size > 0 else 'N/A'}

"""
        
        # 安全地添加整体性能
        if 'error' in metrics:
            report += f"错误: {metrics['error']}\n"
        elif all(k in metrics for k in ['overall_mse', 'overall_mae', 'overall_rmse', 'overall_r2']):
            report += f"""整体性能 (归一化尺度):
- 均方误差 (MSE): {metrics['overall_mse']:.4f}
- 平均绝对误差 (MAE): {metrics['overall_mae']:.4f}
- 均方根误差 (RMSE): {metrics['overall_rmse']:.4f}
- R²分数: {metrics['overall_r2']:.4f}

"""
            
            # 添加原始尺度的指标
            if any(k in metrics for k in ['overall_mse_original', 'overall_mae_original', 'overall_rmse_original', 'overall_r2_original']):
                report += f"""整体性能 (原始尺度):
- 均方误差 (MSE): {metrics.get('overall_mse_original', 'N/A')}
- 平均绝对误差 (MAE): {metrics.get('overall_mae_original', 'N/A')}
- 均方根误差 (RMSE): {metrics.get('overall_rmse_original', 'N/A')}
- R²分数: {metrics.get('overall_r2_original', 'N/A')}

"""
        
        # 安全地添加领域性能
        if 'domain_metrics' in metrics and metrics['domain_metrics']:
            report += "领域性能:\n"
            for domain, domain_metric in metrics['domain_metrics'].items():
                report += f"""
{domain} (归一化尺度):
- 均方误差 (MSE): {domain_metric['mse']:.4f}
- 平均绝对误差 (MAE): {domain_metric['mae']:.4f}
- 均方根误差 (RMSE): {domain_metric['rmse']:.4f}
- R²分数: {domain_metric['r2']:.4f}
- 样本数: {domain_metric['samples']}
"""
                
                # 添加原始尺度的域指标
                if any(k in domain_metric for k in ['mse_original', 'mae_original', 'rmse_original', 'r2_original']):
                    report += f"""
{domain} (原始尺度):
- 均方误差 (MSE): {domain_metric.get('mse_original', 'N/A')}
- 平均绝对误差 (MAE): {domain_metric.get('mae_original', 'N/A')}
- 均方根误差 (RMSE): {domain_metric.get('rmse_original', 'N/A')}
- R²分数: {domain_metric.get('r2_original', 'N/A')}
"""
            
            if 'domain_gap' in metrics:
                report += f"\n领域泛化差距 (MAE, 归一化尺度): {metrics['domain_gap']:.4f}\n"
            if 'domain_gap_original' in metrics:
                report += f"领域泛化差距 (MAE, 原始尺度): {metrics['domain_gap_original']:.4f}\n"
            if 'num_domains' in metrics:
                report += f"领域数量: {metrics['num_domains']}\n"
        else:
            report += "领域性能: 无法计算（数据不足或评估失败）\n"
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        print(report)
        return report
    
    def compare_models(self, model_results: Dict[str, Dict], save_path=None):
        """比较多个模型"""
        comparison_data = []
        for model_name, results in model_results.items():
            metrics = results.get('metrics', {})
            
            # 如果有基本的指标就包含进来，缺失的用N/A填充
            if metrics and ('overall_mse' in metrics or 'overall_mae' in metrics):
                comparison_data.append({
                    'Model': model_name,
                    'Overall MSE': metrics.get('overall_mse', 'N/A'),
                    'Overall MAE': metrics.get('overall_mae', 'N/A'),
                    'Overall RMSE': metrics.get('overall_rmse', 'N/A'),
                    'Overall R²': metrics.get('overall_r2', 'N/A'),
                    'Domain Gap': metrics.get('domain_gap', 'N/A'),
                    'Num Domains': metrics.get('num_domains', 'N/A'),
                    'Test Samples': len(results.get('predictions', []))
                })
            else:
                print(f"Warning: 模型 {model_name} 缺少基本评估指标，将显示为评估失败。")
                comparison_data.append({
                    'Model': model_name,
                    'Overall MSE': 'Failed',
                    'Overall MAE': 'Failed',
                    'Overall RMSE': 'Failed',
                    'Overall R²': 'Failed',
                    'Domain Gap': 'Failed',
                    'Num Domains': 'Failed',
                    'Test Samples': len(results.get('predictions', []))
                })
                
        if not comparison_data:
            print("Warning: 没有任何模型数据可比较。")
            return None
        df = pd.DataFrame(comparison_data)
        
        # 绘制比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 整体MSE比较 - 只绘制有效数据
        valid_mse = df[pd.to_numeric(df['Overall MSE'], errors='coerce').notna()]
        if not valid_mse.empty:
            axes[0, 0].bar(valid_mse['Model'], pd.to_numeric(valid_mse['Overall MSE']))
            axes[0, 0].set_title('Overall MSE Comparison')
            axes[0, 0].set_ylabel('Mean Squared Error')
        else:
            axes[0, 0].text(0.5, 0.5, 'No valid MSE data', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Overall MSE Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 整体MAE比较 - 只绘制有效数据
        valid_mae = df[pd.to_numeric(df['Overall MAE'], errors='coerce').notna()]
        if not valid_mae.empty:
            axes[0, 1].bar(valid_mae['Model'], pd.to_numeric(valid_mae['Overall MAE']))
            axes[0, 1].set_title('Overall MAE Comparison')
            axes[0, 1].set_ylabel('Mean Absolute Error')
        else:
            axes[0, 1].text(0.5, 0.5, 'No valid MAE data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Overall MAE Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 领域差距比较 - 只绘制有效数据
        valid_gap = df[pd.to_numeric(df['Domain Gap'], errors='coerce').notna()]
        if not valid_gap.empty:
            axes[1, 0].bar(valid_gap['Model'], pd.to_numeric(valid_gap['Domain Gap']))
            axes[1, 0].set_title('Domain Generalization Gap')
            axes[1, 0].set_ylabel('Gap')
        else:
            axes[1, 0].text(0.5, 0.5, 'No valid domain gap data', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Domain Generalization Gap')
        axes[1, 0].tick_params(axis='x', rotation=45)
        # 表格
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return df 