#!/usr/bin/env python3
"""
领域泛化实验运行器

此脚本演示如何使用领域泛化框架
在域外数据上比较不同的模型架构。

use adain
"""

import os
import sys
import argparse
from domain_generalization import (
    DomainGeneralizationPipeline, 
    create_experiment_config, 
    set_random_seed,
    print_experiment_summary
)


def main():
    parser = argparse.ArgumentParser(description='领域泛化实验')
    parser.add_argument('--config', type=str, help='实验配置文件路径')
    parser.add_argument('--adain', action='store_true', help='Enable AdaIN')
    parser.add_argument('--models', nargs='+', 
                       choices=['transformer', 'graph_transformer', 'lstm'],
                       default=['transformer', 'graph_transformer', 'lstm'],
                       help='要测试的模型')
    parser.add_argument('--output-dir', type=str, default='results_adain-bpi15-t4',
                       help='结果输出目录')
    parser.add_argument('--num-epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='学习率')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--use-synthetic', action='store_true', default=False,
                       help='使用合成数据进行测试')
    parser.add_argument('--data-path', type=str,
                       help='真实数据文件路径（支持CSV、JSON、JSONL格式）')
    parser.add_argument('--train-domains', nargs='+', type=int, default=[0, 1, 2, 3],
                       help='训练域列表')
    parser.add_argument('--val-domains', nargs='+', type=int, default=[0, 1, 2, 3],
                       help='验证域列表')
    parser.add_argument('--test-domains', nargs='+', type=int, default=[4],
                       help='测试域列表')
    
    args = parser.parse_args()

    if args.adain:
        print("adain is enabled.")
        adain_flag = True
    else:
        print("adain is disabled.")
        adain_flag = False
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 创建实验配置
    if args.config and os.path.exists(args.config):
        print(f"从 {args.config} 加载配置")
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        print("创建默认配置")
        
        # 数据配置
        if args.use_synthetic:
            # 合成数据配置
            data_config = {
                'use_synthetic': True,
                'num_samples': 1000,
                'seq_len': 50,
                'input_dim': 10,
                'num_domains': 5,  # 包含域0-4
                'num_classes': 5,
                'train_domains': args.train_domains,
                'val_domains': args.val_domains,
                'test_domains': args.test_domains,
                'random_split': False  # 使用域分割
            }
        else:
            # 真实数据配置
            data_config = {
                'use_synthetic': False,
                'seq_len': 50,
                'input_dim': 10,  # 将根据CSV文件自动调整
                'train_domains': args.train_domains,
                'val_domains': args.val_domains,
                'test_domains': args.test_domains,
                'random_split': False # 使用域分割进行领域泛化
                
            }
            
            if args.data_path:
                data_config['data_path'] = args.data_path
            else:
                raise ValueError("使用真实数据时必须指定 --data-path 参数")
        
        # 训练配置
        training_config = {
            'batch_size': args.batch_size,
            'learning_rate': 1e-6,  # 较低的学习率
            'weight_decay': 1e-6,   # 较低的权重衰减
            'num_epochs': args.num_epochs,
            'early_stopping_patience': 15,
            'seed': args.seed
        }
        
        config = create_experiment_config(
            model_types=args.models,
            data_config=data_config,
            training_config=training_config,
            adain_flag=adain_flag
        )
    
    # 更新输出目录
    config['output_dir'] = args.output_dir
    # config['use_adain'] = True
    # config['adain_prob'] = 0.8
    
    print("实验配置:")
    print(f"  模型: {list(config['model_configs'].keys())}")
    print(f"  输出目录: {config['output_dir']}")
    print(f"  数据配置: {config['data_config']}")
    print(f"  训练域: {config['data_config']['train_domains']}")
    print(f"  验证域: {config['data_config']['val_domains']}")
    print(f"  测试域: {config['data_config']['test_domains']}")
    print(f"  训练配置: {config['model_configs'][list(config['model_configs'].keys())[0]]}")
    
    # 创建并运行流水线
    pipeline = DomainGeneralizationPipeline(config)
    
    try:
        pipeline.run_experiment(
            data_config=config['data_config'],
            model_configs=config['model_configs']
        )
        
        # 打印摘要
        print_experiment_summary(pipeline.results)
        
        # 获取最佳模型
        best_model, best_score = pipeline.get_best_model()
        print(f"\n🎉 性能最佳的模型: {best_model} (MAE: {best_score:.4f})")
        
        print(f"\n📁 结果已保存到: {config['output_dir']}")
        print("   - 各个模型的结果在子目录中")
        print("   - 模型比较图表和CSV文件")
        print("   - 详细结果以JSON格式保存")
        
    except Exception as e:
        print(f"❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 
 