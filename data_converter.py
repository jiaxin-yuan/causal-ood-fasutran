#!/usr/bin/env python3
"""
数据格式转换工具

此脚本帮助用户将常见格式的数据转换为域泛化项目所需的格式。
支持从CSV、Excel、NumPy数组等格式转换。
"""

import json
import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple, Optional
import pm4py


class DataConverter:
    """数据格式转换器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.domain_encoder = LabelEncoder()
    
    def convert_csv_to_json(self, csv_path: str, output_path: str, 
                           feature_cols: List[str], label_col: str, 
                           domain_col: Optional[str] = None,
                           normalize: bool = True):
        """将CSV文件转换为JSON格式"""
        
        print(f"正在读取CSV文件: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 自动编码所有非数值型特征列
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = LabelEncoder().fit_transform(df[col])
        # 提取特征
        features = df[feature_cols].values
        
        # 处理标签
        labels = df[label_col].values
        if normalize:
            labels = self.label_encoder.fit_transform(labels)
        
        # 处理领域
        if domain_col:
            domains = df[domain_col].values
            if normalize:
                domains = self.domain_encoder.fit_transform(domains)
        else:
            # 如果没有领域列，创建默认领域
            domains = np.zeros(len(features), dtype=int)
        
        # 标准化特征
        if normalize:
            features = self.scaler.fit_transform(features)
        
        # 保存为JSON格式
        data_dict = {
            'data': features.tolist(),
            'labels': labels.tolist(),
            'domains': domains.tolist()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        
        print(f"数据已保存到: {output_path}")
        print(f"数据统计:")
        print(f"  样本数量: {len(features)}")
        print(f"  特征维度: {features.shape[1]}")
        print(f"  类别数量: {len(np.unique(labels))}")
        print(f"  领域数量: {len(np.unique(domains))}")
        
        return data_dict
    
    def convert_csv_to_jsonl(self, csv_path: str, output_path: str,
                            feature_cols: List[str], label_col: str,
                            domain_col: Optional[str] = None,
                            normalize: bool = True):
        """将CSV文件转换为JSONL格式"""
        
        print(f"正在读取CSV文件: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 自动编码所有非数值型特征列
        for col in feature_cols:
            if df[col].dtype == 'object':
                df[col] = LabelEncoder().fit_transform(df[col])
        # 提取特征
        features = df[feature_cols].values
        
        # 处理标签
        labels = df[label_col].values
        if normalize:
            labels = self.label_encoder.fit_transform(labels)
        
        # 处理领域
        if domain_col:
            domains = df[domain_col].values
            if normalize:
                domains = self.domain_encoder.fit_transform(domains)
        else:
            domains = np.zeros(len(features), dtype=int)
        
        # 标准化特征
        if normalize:
            features = self.scaler.fit_transform(features)
        
        # 保存为JSONL格式
        with open(output_path, 'w', encoding='utf-8') as f:
            for i in range(len(features)):
                sample = {
                    'data': features[i].tolist(),
                    'label': int(labels[i]),
                    'domain': int(domains[i])
                }
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"数据已保存到: {output_path}")
        print(f"数据统计:")
        print(f"  样本数量: {len(features)}")
        print(f"  特征维度: {features.shape[1]}")
        print(f"  类别数量: {len(np.unique(labels))}")
        print(f"  领域数量: {len(np.unique(domains))}")
    
    def convert_numpy_to_json(self, data_path: str, output_path: str,
                             label_path: Optional[str] = None,
                             domain_path: Optional[str] = None,
                             normalize: bool = True):
        """将NumPy数组转换为JSON格式"""
        
        print(f"正在读取NumPy数据: {data_path}")
        features = np.load(data_path)
        
        # 处理标签
        if label_path:
            labels = np.load(label_path)
        else:
            labels = np.zeros(len(features), dtype=int)
        
        # 处理领域
        if domain_path:
            domains = np.load(domain_path)
        else:
            domains = np.zeros(len(features), dtype=int)
        
        # 标准化
        if normalize:
            features = self.scaler.fit_transform(features)
            if label_path:
                labels = self.label_encoder.fit_transform(labels)
            if domain_path:
                domains = self.domain_encoder.fit_transform(domains)
        
        # 保存为JSON格式
        data_dict = {
            'data': features.tolist(),
            'labels': labels.tolist(),
            'domains': domains.tolist()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=2, ensure_ascii=False)
        
        print(f"数据已保存到: {output_path}")
        return data_dict
    
    def create_config_template(self, data_path: str, output_path: str):
        """根据数据文件创建配置模板"""
        
        print(f"正在分析数据文件: {data_path}")
        
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            
            features = np.array(data_dict['data'])
            labels = np.array(data_dict['labels'])
            domains = np.array(data_dict['domains'])
            
        elif data_path.endswith('.jsonl'):
            features = []
            labels = []
            domains = []
            
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line.strip())
                    features.append(sample['data'])
                    labels.append(sample['label'])
                    domains.append(sample['domain'])
            
            features = np.array(features)
            labels = np.array(labels)
            domains = np.array(domains)
        
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")
        
        # 创建配置模板
        unique_domains = np.unique(domains)
        unique_labels = np.unique(labels)
        
        config = {
            "data_config": {
                "use_synthetic": False,
                "data_path": data_path,
                "num_samples": len(features),
                "seq_len": features.shape[1] if len(features.shape) > 1 else 1,
                "input_dim": features.shape[1] if len(features.shape) > 1 else features.shape[0],
                "num_domains": len(unique_domains),
                "num_classes": len(unique_labels),
                "train_domains": [int(unique_domains[0])],
                "val_domains": [int(unique_domains[1])] if len(unique_domains) > 1 else [int(unique_domains[0])],
                "test_domains": [int(unique_domains[2])] if len(unique_domains) > 2 else [int(unique_domains[-1])]
            },
            "model_configs": {
                "transformer": {
                    "type": "transformer",
                    "input_dim": features.shape[1] if len(features.shape) > 1 else features.shape[0],
                    "d_model": 256,
                    "nhead": 8,
                    "num_layers": 4,
                    "output_dim": len(unique_labels),
                    "dropout": 0.1,
                    "batch_size": 32,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-5,
                    "num_epochs": 50,
                    "early_stopping_patience": 10,
                    "seed": 42
                }
            },
            "output_dir": "results"
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"配置模板已保存到: {output_path}")
        print(f"数据统计:")
        print(f"  样本数量: {len(features)}")
        print(f"  特征维度: {features.shape[1] if len(features.shape) > 1 else features.shape[0]}")
        print(f"  类别数量: {len(unique_labels)}")
        print(f"  领域数量: {len(unique_domains)}")
        print(f"  领域ID: {unique_domains.tolist()}")
        print(f"  标签ID: {unique_labels.tolist()}")

    def convert_xes_to_json(self, xes_path: str, output_path: str):
        """将 XES 文件转换为 JSON 格式"""
        log = pm4py.read_xes(xes_path)
        data, labels, domains = [], [], []
        for trace in log:
            features = []
            for event in trace:
                if isinstance(event, dict):
                    features.append([event.get(attr, 0) for attr in event.keys()])
                else:
                    print(f"Warning: event is not a dict, but {type(event)}: {event}")
                    continue
            data.append(features)
            # 这里 label/domain 需根据实际业务逻辑调整
            labels.append(0)
            domains.append(0)
        data_dict = {'data': data, 'labels': labels, 'domains': domains}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, ensure_ascii=False)

    def convert_xes_to_jsonl(self, xes_path: str, output_path: str):
        """将 XES 文件转换为 JSONL 格式"""
        log = pm4py.read_xes(xes_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for trace in log:
                features = []
                for event in trace:
                    if isinstance(event, dict):
                        features.append([event.get(attr, 0) for attr in event.keys()])
                    else:
                        print(f"Warning: event is not a dict, but {type(event)}: {event}")
                        continue
                # 这里 label/domain 需根据实际业务逻辑调整
                item = {'data': features, 'label': 0, 'domain': 0}
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def convert_xes_to_suffix_jsonl(self, xes_path: str, output_path: str, min_prefix: int = 1, min_suffix: int = 1):
        """将 XES 文件转换为 Suffix Prediction 任务的 JSONL 格式"""
        log = pm4py.read_xes(xes_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for trace in log:
                events = [event for event in trace]
                L = len(events)
                for split in range(min_prefix, L - min_suffix + 1):
                    prefix = [dict(event) for event in events[:split] if isinstance(event, dict)]
                    suffix = [dict(event) for event in events[split:] if isinstance(event, dict)]
                    # 打印警告
                    for event in events[:split]:
                        if not isinstance(event, dict):
                            print(f"Warning: event in prefix is not a dict, but {type(event)}: {event}")
                    for event in events[split:]:
                        if not isinstance(event, dict):
                            print(f"Warning: event in suffix is not a dict, but {type(event)}: {event}")
                    item = {'data': prefix, 'label': suffix}
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='数据格式转换工具')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--format', choices=['json', 'jsonl'], default='json', help='输出格式')
    parser.add_argument('--feature-cols', nargs='+', help='特征列名（CSV文件）')
    parser.add_argument('--label-col', type=str, help='标签列名（CSV文件）')
    parser.add_argument('--domain-col', type=str, help='领域列名（CSV文件）')
    parser.add_argument('--label-path', type=str, help='标签文件路径（NumPy文件）')
    parser.add_argument('--domain-path', type=str, help='领域文件路径（NumPy文件）')
    parser.add_argument('--no-normalize', action='store_true', help='不进行标准化')
    parser.add_argument('--create-config', type=str, help='创建配置模板文件路径')
    
    args = parser.parse_args()
    
    converter = DataConverter()
    normalize = not args.no_normalize
    
    # 根据文件扩展名选择转换方法
    if args.input.endswith('.csv'):
        if not args.feature_cols or not args.label_col:
            print("错误: CSV文件需要指定 --feature-cols 和 --label-col")
            return
        
        if args.format == 'json':
            converter.convert_csv_to_json(
                args.input, args.output, args.feature_cols, 
                args.label_col, args.domain_col, normalize
            )
        else:
            converter.convert_csv_to_jsonl(
                args.input, args.output, args.feature_cols,
                args.label_col, args.domain_col, normalize
            )
    
    elif args.input.endswith('.npy'):
        converter.convert_numpy_to_json(
            args.input, args.output, args.label_path,
            args.domain_path, normalize
        )
    
    elif args.input.endswith(('.json', '.jsonl')):
        if args.create_config:
            converter.create_config_template(args.input, args.create_config)
        else:
            print("对于JSON/JSONL文件，请使用 --create-config 创建配置模板")
    
    elif args.input.endswith('.xes'):
        converter.convert_xes_to_json(args.input, args.output)
        converter.convert_xes_to_jsonl(args.input, args.output.replace('.json', '.jsonl'))
        converter.convert_xes_to_suffix_jsonl(args.input, args.output.replace('.json', '.jsonl'))
    
    else:
        print(f"不支持的文件格式: {args.input}")
        print("支持的文件格式: .csv, .npy, .json, .jsonl, .xes")


if __name__ == "__main__":
    main() 