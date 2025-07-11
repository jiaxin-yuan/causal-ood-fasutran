import torch
import numpy as np
import json
import os
import pandas as pd  # 添加pandas支持
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Dict, List, Tuple, Optional
import random
from sklearn.preprocessing import LabelEncoder  # 添加sklearn支持


class DomainDataset(Dataset):
    """用于领域泛化的数据集类"""
    
    def __init__(self, data, labels, domains, transform=None):
        self.data = data
        self.labels = labels
        self.domains = domains
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取原始数据
        data_sample = self.data[idx]
        label_sample = self.labels[idx]
        domain_sample = self.domains[idx]
        
        # 检查并处理异常值
        if np.isnan(data_sample).any() or np.isinf(data_sample).any():
            print(f"警告：样本 {idx} 的数据包含异常值，用零替换")
            data_sample = np.nan_to_num(data_sample, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(label_sample) or np.isinf(label_sample):
            print(f"警告：样本 {idx} 的标签包含异常值，用零替换")
            label_sample = 0.0
        
        sample = {
            'data': data_sample,
            'rtime': float(label_sample),
            'domain': int(domain_sample)
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class DataLoader:
    """用于领域泛化实验的数据加载器"""
    
    def __init__(self, config):
        self.config = config
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        self.seed = config.get('seed', 42)
        
        # 设置随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def generate_synthetic_data(self, num_samples=1000, seq_len=50, input_dim=128, 
                               num_domains=3, num_classes=5):
        """生成用于测试的合成数据"""
        
        data = []
        labels = []
        domains = []
        
        for domain in range(num_domains):
            # 生成领域特定的数据
            domain_data = np.random.randn(num_samples, seq_len, input_dim)
            
            # 添加领域特定的模式
            domain_data += domain * 0.5  # 领域偏移
            
            # 生成标签
            domain_labels = np.random.randint(0, num_classes, num_samples)
            
            data.extend(domain_data)
            labels.extend(domain_labels)
            domains.extend([domain] * num_samples)
        
        return np.array(data), np.array(labels), np.array(domains)
    
    def split_data_by_domain(self, data, labels, domains, train_domains, val_domains, test_domains, random_split=False):
        """按领域分割数据用于领域泛化"""
        
        # 打印域分布信息
        from collections import Counter
        domain_counts = Counter(domains)
        unique_domains = sorted(domain_counts.keys())
        print(f"数据中的域分布:")
        for domain in unique_domains:
            count = domain_counts[domain]
            percentage = (count / len(domains)) * 100
            print(f"  域 {domain}: {count:,} 样本 ({percentage:.2f}%)")
        
        print(f"分割配置:")
        print(f"  训练域: {train_domains}")
        print(f"  验证域: {val_domains}")
        print(f"  测试域: {test_domains}")
        
        # 如果启用随机分割模式（用于单域数据）
        if random_split or len(np.unique(domains)) == 1:
            print("使用随机分割模式（单域数据）")
            n_samples = len(data)
            indices = np.random.permutation(n_samples)
            
            # 按70-15-15比例分割训练、验证、测试集
            train_size = int(0.7 * n_samples)
            val_size = int(0.15 * n_samples)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            train_data = data[train_indices]
            train_labels = labels[train_indices]
            train_domains = domains[train_indices]
            
            val_data = data[val_indices]
            val_labels = labels[val_indices]
            val_domains = domains[val_indices]
            
            test_data = data[test_indices]
            test_labels = labels[test_indices]
            test_domains = domains[test_indices]
            
            return (train_data, train_labels, train_domains), \
                   (val_data, val_labels, val_domains), \
                   (test_data, test_labels, test_domains)
        
        # 领域泛化的域分割逻辑
        print("使用领域泛化分割模式")
        
        # 测试集：使用指定的测试领域
        test_mask = np.isin(domains, test_domains)
        test_data = data[test_mask]
        test_labels = labels[test_mask]
        test_domains_array = domains[test_mask]
        
        # 检查测试域是否存在数据
        if len(test_data) == 0:
            available_domains = list(unique_domains)
            print(f"⚠️  测试域{test_domains}没有数据！")
            print(f"可用的域: {available_domains}")
            raise ValueError(f"测试域{test_domains}在数据中不存在，可用域: {available_domains}")
        
        # 如果训练域和验证域相同（leave-one-domain-out设置）
        if set(train_domains) == set(val_domains):
            print("训练域和验证域相同，使用leave-one-domain-out策略")
            # 从训练域中获取所有数据
            train_val_mask = np.isin(domains, train_domains)
            train_val_data = data[train_val_mask]
            train_val_labels = labels[train_val_mask]
            train_val_domains = domains[train_val_mask]
            
            # 检查训练域是否存在数据
            if len(train_val_data) == 0:
                print(f"⚠️  训练域{train_domains}没有数据！")
                raise ValueError(f"训练域{train_domains}在数据中不存在")
            
            # 按80-20比例分割训练和验证集
            n_samples = len(train_val_data)
            indices = np.random.permutation(n_samples)
            train_size = int(0.8 * n_samples)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            train_data = train_val_data[train_indices]
            train_labels = train_val_labels[train_indices]
            train_domains_array = train_val_domains[train_indices]
            
            val_data = train_val_data[val_indices]
            val_labels = train_val_labels[val_indices]
            val_domains_array = train_val_domains[val_indices]
        else:
            print("训练域和验证域不同，使用独立域分割")
            # 使用不同的领域作为训练和验证集
            train_mask = np.isin(domains, train_domains)
            val_mask = np.isin(domains, val_domains)
            
            train_data = data[train_mask]
            train_labels = labels[train_mask]
            train_domains_array = domains[train_mask]
            
            val_data = data[val_mask]
            val_labels = labels[val_mask]
            val_domains_array = domains[val_mask]
            
            # 检查训练域和验证域是否存在数据
            if len(train_data) == 0:
                print(f"⚠️  训练域{train_domains}没有数据！")
                raise ValueError(f"训练域{train_domains}在数据中不存在")
            
            if len(val_data) == 0:
                print(f"⚠️  验证域{val_domains}没有数据！")
                raise ValueError(f"验证域{val_domains}在数据中不存在")
        
        print(f"领域泛化分割结果:")
        print(f"  训练集: {len(train_data)} 样本，域分布: {Counter(train_domains_array)}")
        print(f"  验证集: {len(val_data)} 样本，域分布: {Counter(val_domains_array)}")
        print(f"  测试集: {len(test_data)} 样本，域分布: {Counter(test_domains_array)}")
        
        return (train_data, train_labels, train_domains_array), \
               (val_data, val_labels, val_domains_array), \
               (test_data, test_labels, test_domains_array)
    
    def create_dataloaders(self, train_data, val_data, test_data):
        """创建PyTorch数据加载器"""
        
        train_dataset = DomainDataset(
            train_data[0], train_data[1], train_data[2]
        )
        val_dataset = DomainDataset(
            val_data[0], val_data[1], val_data[2]
        )
        test_dataset = DomainDataset(
            test_data[0], test_data[1], test_data[2]
        )
        
        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        val_loader = TorchDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def load_real_data(self, data_path):
        """从文件加载真实数据"""
        if data_path.endswith('.jsonl'):
            return self._load_jsonl_data(data_path)
        elif data_path.endswith('.json'):
            return self._load_json_data(data_path)
        elif data_path.endswith('.csv'):
            return self._load_csv_data(data_path)
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
    
    def load_real_data_with_domain_normalization(self, data_path, train_domains, val_domains, test_domains):
        """从文件加载真实数据，并基于训练域进行归一化"""
        # 先加载原始数据（不进行归一化）
        if data_path.endswith('.jsonl'):
            data, labels, domains = self._load_jsonl_data_raw(data_path)
        elif data_path.endswith('.json'):
            data, labels, domains = self._load_json_data_raw(data_path)
        elif data_path.endswith('.csv'):
            data, labels, domains = self._load_csv_data_raw(data_path)
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
        
        # 基于训练域计算归一化参数
        train_mask = np.isin(domains, train_domains)
        train_rtime = labels[train_mask]
        
        if len(train_rtime) == 0:
            raise ValueError(f"训练域 {train_domains} 没有找到数据")
        
        # 计算训练域的 rtime 统计信息
        rtime_min_train = float(train_rtime.min())
        rtime_max_train = float(train_rtime.max())
        rtime_mean_train = float(train_rtime.mean())
        rtime_std_train = float(train_rtime.std())
        
        print(f"训练域 rtime 统计信息:")
        print(f"  最小值: {rtime_min_train:.4f}")
        print(f"  最大值: {rtime_max_train:.4f}")
        print(f"  均值: {rtime_mean_train:.4f}")
        print(f"  标准差: {rtime_std_train:.4f}")
        
        # 使用训练域的统计信息对所有数据进行归一化
        if rtime_std_train > 1e-8:
            # 使用标准化 (z-score)
            labels_normalized = (labels - rtime_mean_train) / rtime_std_train
            normalization_method = "标准化"
        else:
            # 使用最小-最大归一化
            if rtime_max_train - rtime_min_train > 1e-8:
                labels_normalized = (labels - rtime_min_train) / (rtime_max_train - rtime_min_train)
                labels_normalized = labels_normalized * 2 - 1  # 调整到[-1, 1]范围
                normalization_method = "最小-最大归一化"
            else:
                print("警告：训练域rtime值完全相同，设置为零")
                labels_normalized = np.zeros_like(labels)
                normalization_method = "零值填充"
        
        print(f"使用 {normalization_method} 对所有数据进行归一化")
        print(f"归一化后 rtime 范围: {labels_normalized.min():.4f} - {labels_normalized.max():.4f}")
        
        # 保存归一化参数以便后续使用
        self.normalization_params = {
            'rtime_min_train': rtime_min_train,
            'rtime_max_train': rtime_max_train,
            'rtime_mean_train': rtime_mean_train,
            'rtime_std_train': rtime_std_train,
            'method': normalization_method
        }
        
        return data, labels_normalized, domains
    
    def inverse_normalize_rtime(self, normalized_values):
        """将归一化后的rtime值转换回原始尺度"""
        if not hasattr(self, 'normalization_params') or not self.normalization_params:
            print("警告：没有找到归一化参数，返回原始值")
            return normalized_values
        
        params = self.normalization_params
        method = params['method']
        
        if method == "标准化":
            # 反标准化: x = (x_norm * std) + mean
            original_values = (normalized_values * params['rtime_std_train']) + params['rtime_mean_train']
        elif method == "最小-最大归一化":
            # 反最小-最大归一化: x = ((x_norm + 1) / 2) * (max - min) + min
            original_values = ((normalized_values + 1) / 2) * (params['rtime_max_train'] - params['rtime_min_train']) + params['rtime_min_train']
        elif method == "零值填充":
            # 如果原始值都相同，返回训练域的均值
            original_values = np.full_like(normalized_values, params['rtime_mean_train'])
        else:
            print(f"警告：未知的归一化方法 '{method}'，返回原始值")
            original_values = normalized_values
        
        return original_values
    
    def get_normalization_info(self):
        """获取归一化参数信息"""
        if hasattr(self, 'normalization_params') and self.normalization_params:
            return self.normalization_params.copy()
        else:
            return None
    
    def _load_jsonl_data(self, data_path):
        """从JSONL文件加载数据"""
        data = []
        labels = []
        domains = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item['data'])
                labels.append(item['label'])
                domains.append(item['domain'])
        
        return np.array(data), np.array(labels), np.array(domains)
    
    def _load_json_data(self, data_path):
        """从JSON文件加载数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        
        # 支持两种数据键名格式
        if 'ori_data' in data_dict:
            data_key = 'ori_data'
        elif 'data' in data_dict:
            data_key = 'data'
        else:
            raise KeyError("数据文件中必须包含 'data' 或 'ori_data' 键")
        
        data = np.array(data_dict[data_key])
        rtime = np.array(data_dict['labels'])
        domains = np.array(data_dict['domains'])
        
        # 检查数据中的异常值
        print(f"原始数据形状: {data.shape}")
        print(f"原始rtime范围: {rtime.min():.2f} - {rtime.max():.2f}")
        print(f"data中是否有nan: {np.isnan(data).any()}")
        print(f"data中是否有inf: {np.isinf(data).any()}")
        print(f"rtime中是否有nan: {np.isnan(rtime).any()}")
        print(f"rtime中是否有inf: {np.isinf(rtime).any()}")
        
        # 移除包含异常值的样本
        # 检查data中的异常值
        data_valid_mask = ~(np.isnan(data).any(axis=tuple(range(1, data.ndim))) | 
                           np.isinf(data).any(axis=tuple(range(1, data.ndim))))
        
        # 检查rtime中的异常值
        rtime_valid_mask = ~(np.isnan(rtime) | np.isinf(rtime))
        
        # 综合有效性掩码
        valid_mask = data_valid_mask & rtime_valid_mask
        
        if not valid_mask.all():
            print(f"发现异常值，正在移除...")
            print(f"data异常样本数: {(~data_valid_mask).sum()}")
            print(f"rtime异常样本数: {(~rtime_valid_mask).sum()}")
            
            data = data[valid_mask]
            rtime = rtime[valid_mask]
            domains = domains[valid_mask]
            print(f"移除异常值后样本数: {len(rtime)}")
        
        # 对data进行异常值处理
        # 替换极值为合理范围内的值
        data_flat = data.reshape(-1)
        data_q1 = np.percentile(data_flat, 25)
        data_q3 = np.percentile(data_flat, 75)
        data_iqr = data_q3 - data_q1
        data_lower = data_q1 - 1.5 * data_iqr
        data_upper = data_q3 + 1.5 * data_iqr
        
        # 裁剪极值
        data = np.clip(data, data_lower, data_upper)
        
        # 对data进行标准化
        data_mean = data.mean()
        data_std = data.std()
        if data_std > 1e-8:  # 避免除零
            data = (data - data_mean) / data_std
        else:
            print("警告：data标准差接近0，跳过标准化")
        
        # 对rtime进行稳健的归一化
        rtime_mean = rtime.mean()
        rtime_std = rtime.std()
        
        print(f"rtime统计: 均值={rtime_mean:.4f}, 标准差={rtime_std:.4f}")
        
        if rtime_std > 1e-8:  # 避免除零
            rtime = (rtime - rtime_mean) / rtime_std
        else:
            print("警告：rtime标准差接近0，使用最小-最大归一化")
            rtime_min, rtime_max = rtime.min(), rtime.max()
            if rtime_max - rtime_min > 1e-8:
                rtime = (rtime - rtime_min) / (rtime_max - rtime_min)
                # 将范围调整到[-1, 1]
                rtime = rtime * 2 - 1
            else:
                print("警告：rtime值完全相同，设置为零")
                rtime = np.zeros_like(rtime)
        
        print(f"归一化后rtime范围: {rtime.min():.4f} - {rtime.max():.4f}")
        
        # 最终检查是否还有异常值
        if np.isnan(data).any() or np.isinf(data).any():
            print("警告：data标准化后仍有异常值，用零替换")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(rtime).any() or np.isinf(rtime).any():
            print("警告：rtime标准化后仍有异常值，用零替换")
            rtime = np.nan_to_num(rtime, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 记录原始形状
        original_shape = data.shape
        # 如果数据是2D的 [batch_size, features]，转换为3D [batch_size, seq_len, features]
        if len(data.shape) == 2:
            seq_len = self.config.get('seq_len', 50)
            features = data.shape[1]
            data_reshaped = []
            for sample in data:
                sample_seq = np.tile(sample, (seq_len, 1))
                data_reshaped.append(sample_seq)
            data = np.array(data_reshaped)
            print(f"数据已从 {original_shape} 重塑为 {data.shape}")
        
        # 最终验证
        print(f"最终data形状: {data.shape}")
        print(f"最终data范围: {data.min():.4f} - {data.max():.4f}")
        print(f"最终rtime范围: {rtime.min():.4f} - {rtime.max():.4f}")
        print(f"最终检查 - data中是否有nan: {np.isnan(data).any()}")
        print(f"最终检查 - rtime中是否有nan: {np.isnan(rtime).any()}")
        
        return data, rtime, domains
    
    def _load_csv_data(self, data_path):
        """从CSV文件加载数据"""
        print(f"从CSV文件加载数据: {data_path}")
        
        # 读取CSV文件
        df = pd.read_csv(data_path)
        print(f"CSV数据形状: {df.shape}")
        print(f"CSV列名: {list(df.columns)}")
        
        # 检查必要的列
        required_columns = ['rtime', 'domain']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
        
        # 提取标签和域
        rtime = df['rtime'].values
        domains = df['domain'].values
        
        # 提取特征（除了rtime和domain的其他列）
        feature_columns = [col for col in df.columns if col not in ['rtime', 'domain']]
        
        print(f"特征列: {feature_columns}")
        print(f"特征数量: {len(feature_columns)}")
        
        # 处理特征数据 - 将非数值列转换为数值
        processed_features = []
        for col in feature_columns:
            col_data = df[col]
            if col_data.dtype == 'object':  # 字符串类型
                # 使用标签编码将字符串转换为数值
                le = LabelEncoder()
                col_data = le.fit_transform(col_data.astype(str))
                print(f"列 '{col}' 从字符串转换为数值 (唯一值: {len(le.classes_)})")
            else:
                col_data = col_data.values
            processed_features.append(col_data)
        
        # 转换为numpy数组
        data = np.column_stack(processed_features).astype(np.float32)
        
        print(f"数据形状: {data.shape}")
        print(f"标签范围: {rtime.min():.2f} - {rtime.max():.2f}")
        
        # 域分布统计
        from collections import Counter
        domain_counts = Counter(domains)
        print(f"域分布:")
        for domain, count in sorted(domain_counts.items()):
            percentage = (count / len(domains)) * 100
            print(f"  域 {domain}: {count:,} 样本 ({percentage:.2f}%)")
        
        # 检查数据中的异常值
        print(f"data中是否有nan: {np.isnan(data).any()}")
        print(f"data中是否有inf: {np.isinf(data).any()}")
        print(f"rtime中是否有nan: {np.isnan(rtime).any()}")
        print(f"rtime中是否有inf: {np.isinf(rtime).any()}")
        
        # 移除包含异常值的样本
        data_valid_mask = ~(np.isnan(data).any(axis=1) | np.isinf(data).any(axis=1))
        rtime_valid_mask = ~(np.isnan(rtime) | np.isinf(rtime))
        valid_mask = data_valid_mask & rtime_valid_mask
        
        if not valid_mask.all():
            print(f"发现异常值，正在移除...")
            print(f"data异常样本数: {(~data_valid_mask).sum()}")
            print(f"rtime异常样本数: {(~rtime_valid_mask).sum()}")
            
            data = data[valid_mask]
            rtime = rtime[valid_mask]
            domains = domains[valid_mask]
            print(f"移除异常值后样本数: {len(rtime)}")
        
        # 对data进行异常值处理（使用IQR方法）
        for i in range(data.shape[1]):
            col_data = data[:, i]
            q1 = np.percentile(col_data, 25)
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 裁剪极值
            data[:, i] = np.clip(col_data, lower_bound, upper_bound)
        
        # 对data进行标准化
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        
        # 避免除零
        data_std = np.where(data_std < 1e-8, 1.0, data_std)
        data = (data - data_mean) / data_std
        
        # 对rtime进行标准化
        rtime_mean = rtime.mean()
        rtime_std = rtime.std()
        
        print(f"rtime统计: 均值={rtime_mean:.4f}, 标准差={rtime_std:.4f}")
        
        if rtime_std > 1e-8:
            rtime = (rtime - rtime_mean) / rtime_std
        else:
            print("警告：rtime标准差接近0，使用最小-最大归一化")
            rtime_min, rtime_max = rtime.min(), rtime.max()
            if rtime_max - rtime_min > 1e-8:
                rtime = (rtime - rtime_min) / (rtime_max - rtime_min)
                rtime = rtime * 2 - 1  # 调整到[-1, 1]范围
            else:
                print("警告：rtime值完全相同，设置为零")
                rtime = np.zeros_like(rtime)
        
        print(f"归一化后rtime范围: {rtime.min():.4f} - {rtime.max():.4f}")
        
        # 最终检查异常值
        if np.isnan(data).any() or np.isinf(data).any():
            print("警告：data标准化后仍有异常值，用零替换")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        if np.isnan(rtime).any() or np.isinf(rtime).any():
            print("警告：rtime标准化后仍有异常值，用零替换")
            rtime = np.nan_to_num(rtime, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转换为3D格式 [batch_size, seq_len, features]
        original_shape = data.shape
        if len(data.shape) == 2:
            seq_len = self.config.get('seq_len', 50)
            features = data.shape[1]
            data_reshaped = []
            for sample in data:
                # 将特征重复seq_len次创建序列
                sample_seq = np.tile(sample, (seq_len, 1))
                data_reshaped.append(sample_seq)
            data = np.array(data_reshaped)
            print(f"数据已从 {original_shape} 重塑为 {data.shape}")
        
        # 最终验证
        print(f"最终data形状: {data.shape}")
        print(f"最终data范围: {data.min():.4f} - {data.max():.4f}")
        print(f"最终rtime范围: {rtime.min():.4f} - {rtime.max():.4f}")
        print(f"最终检查 - data中是否有nan: {np.isnan(data).any()}")
        print(f"最终检查 - rtime中是否有nan: {np.isnan(rtime).any()}")
        
        return data, rtime, domains
    
    def _load_jsonl_data_raw(self, data_path):
        """从JSONL文件加载原始数据（不进行归一化）"""
        data = []
        labels = []
        domains = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item['data'])
                labels.append(item['label'])
                domains.append(item['domain'])
        
        return np.array(data), np.array(labels), np.array(domains)
    
    def _load_json_data_raw(self, data_path):
        """从JSON文件加载原始数据（不进行归一化）"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        
        # 支持两种数据键名格式
        if 'ori_data' in data_dict:
            data_key = 'ori_data'
        elif 'data' in data_dict:
            data_key = 'data'
        else:
            raise KeyError("数据文件中必须包含 'data' 或 'ori_data' 键")
        
        data = np.array(data_dict[data_key])
        rtime = np.array(data_dict['labels'])
        domains = np.array(data_dict['domains'])
        
        # 检查数据中的异常值
        print(f"原始数据形状: {data.shape}")
        print(f"原始rtime范围: {rtime.min():.2f} - {rtime.max():.2f}")
        
        # 移除包含异常值的样本
        data_valid_mask = ~(np.isnan(data).any(axis=tuple(range(1, data.ndim))) | 
                           np.isinf(data).any(axis=tuple(range(1, data.ndim))))
        rtime_valid_mask = ~(np.isnan(rtime) | np.isinf(rtime))
        valid_mask = data_valid_mask & rtime_valid_mask
        
        if not valid_mask.all():
            print(f"发现异常值，正在移除...")
            print(f"data异常样本数: {(~data_valid_mask).sum()}")
            print(f"rtime异常样本数: {(~rtime_valid_mask).sum()}")
            
            data = data[valid_mask]
            rtime = rtime[valid_mask]
            domains = domains[valid_mask]
            print(f"移除异常值后样本数: {len(rtime)}")
        
        # 对data进行异常值处理和标准化
        data_flat = data.reshape(-1)
        data_q1 = np.percentile(data_flat, 25)
        data_q3 = np.percentile(data_flat, 75)
        data_iqr = data_q3 - data_q1
        data_lower = data_q1 - 1.5 * data_iqr
        data_upper = data_q3 + 1.5 * data_iqr
        
        data = np.clip(data, data_lower, data_upper)
        
        # 标准化 data
        data_mean = data.mean()
        data_std = data.std()
        if data_std > 1e-8:
            data = (data - data_mean) / data_std
        
        # 最终检查data异常值
        if np.isnan(data).any() or np.isinf(data).any():
            print("警告：data标准化后仍有异常值，用零替换")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 处理数据维度
        if len(data.shape) == 2:
            seq_len = self.config.get('seq_len', 50)
            features = data.shape[1]
            data_reshaped = []
            for sample in data:
                sample_seq = np.tile(sample, (seq_len, 1))
                data_reshaped.append(sample_seq)
            data = np.array(data_reshaped)
            print(f"数据已重塑为: {data.shape}")
        
        return data, rtime, domains
    
    def _load_csv_data_raw(self, data_path):
        """从CSV文件加载原始数据（不进行归一化）"""
        print(f"从CSV文件加载原始数据: {data_path}")
        
        # 读取CSV文件
        df = pd.read_csv(data_path)
        print(f"CSV数据形状: {df.shape}")
        
        # 检查必要的列
        required_columns = ['rtime', 'domain']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要的列: {missing_columns}")
        
        # 提取标签和域
        rtime = df['rtime'].values
        domains = df['domain'].values
        
        # 提取特征
        feature_columns = [col for col in df.columns if col not in ['rtime', 'domain']]
        processed_features = []
        
        for col in feature_columns:
            col_data = df[col]
            if col_data.dtype == 'object':
                le = LabelEncoder()
                col_data = le.fit_transform(col_data.astype(str))
            else:
                col_data = col_data.values
            processed_features.append(col_data)
        
        data = np.column_stack(processed_features).astype(np.float32)
        
        print(f"数据形状: {data.shape}")
        print(f"原始rtime范围: {rtime.min():.2f} - {rtime.max():.2f}")
        
        # 移除异常值
        data_valid_mask = ~(np.isnan(data).any(axis=1) | np.isinf(data).any(axis=1))
        rtime_valid_mask = ~(np.isnan(rtime) | np.isinf(rtime))
        valid_mask = data_valid_mask & rtime_valid_mask
        
        if not valid_mask.all():
            print(f"发现异常值，正在移除...")
            data = data[valid_mask]
            rtime = rtime[valid_mask]
            domains = domains[valid_mask]
            print(f"移除异常值后样本数: {len(rtime)}")
        
        # 对data进行异常值处理和标准化
        for i in range(data.shape[1]):
            col_data = data[:, i]
            q1 = np.percentile(col_data, 25)
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            data[:, i] = np.clip(col_data, lower_bound, upper_bound)
        
        # 标准化data
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        data_std = np.where(data_std < 1e-8, 1.0, data_std)
        data = (data - data_mean) / data_std
        
        # 最终检查data异常值
        if np.isnan(data).any() or np.isinf(data).any():
            print("警告：data标准化后仍有异常值，用零替换")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 转换为3D格式
        if len(data.shape) == 2:
            seq_len = self.config.get('seq_len', 50)
            features = data.shape[1]
            data_reshaped = []
            for sample in data:
                sample_seq = np.tile(sample, (seq_len, 1))
                data_reshaped.append(sample_seq)
            data = np.array(data_reshaped)
            print(f"数据已重塑为: {data.shape}")
        
        return data, rtime, domains 