#!/usr/bin/env python3
"""
AdaIN功能测试脚本

测试AdaIN层的功能，包括：
1. AdaIN层的正确性
2. 模型能否正确使用AdaIN层
3. 训练过程中的AdaIN使用
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from domain_generalization.models import AdaIN, TransformerModel, LSTMModel, GraphTransformer
from domain_generalization.data_loader import DataLoader
from domain_generalization.trainer import Trainer
import json


def test_adain_layer():
    """测试AdaIN层的基本功能"""
    print("="*50)
    print("测试AdaIN层的基本功能")
    print("="*50)
    
    adain = AdaIN(eps=1e-5)
    
    # 测试3D输入 [batch_size, seq_len, feature_dim]
    batch_size, seq_len, feature_dim = 4, 10, 256
    z1 = torch.randn(batch_size, seq_len, feature_dim)
    z2 = torch.randn(batch_size, seq_len, feature_dim)
    
    print(f"输入形状: z1={z1.shape}, z2={z2.shape}")
    
    # 应用AdaIN
    result = adain(z1, z2)
    print(f"输出形状: {result.shape}")
    
    # 验证输出形状
    assert result.shape == z1.shape, f"输出形状不匹配: {result.shape} != {z1.shape}"
    
    # 验证AdaIN公式
    # result应该具有z2的统计特性
    z1_mean = z1.mean(dim=-1, keepdim=True)
    z1_std = z1.std(dim=-1, keepdim=True) + 1e-5
    z2_mean = z2.mean(dim=-1, keepdim=True)
    z2_std = z2.std(dim=-1, keepdim=True) + 1e-5
    
    expected = z2_mean + z2_std * (z1 - z1_mean) / z1_std
    
    # 检查结果是否接近预期
    diff = torch.abs(result - expected).mean()
    print(f"AdaIN结果与预期的平均差异: {diff.item():.8f}")
    assert diff < 1e-5, f"AdaIN结果不正确，差异过大: {diff.item()}"
    
    print("✅ AdaIN层基本功能测试通过")
    
    # 测试2D输入 [batch_size, feature_dim]
    z1_2d = torch.randn(batch_size, feature_dim)
    z2_2d = torch.randn(batch_size, feature_dim)
    
    result_2d = adain(z1_2d, z2_2d)
    print(f"2D输入测试: {z1_2d.shape} -> {result_2d.shape}")
    assert result_2d.shape == z1_2d.shape, "2D输入输出形状不匹配"
    
    print("✅ AdaIN层2D输入测试通过")


def test_models_with_adain():
    """测试模型能否正确使用AdaIN层"""
    print("\n" + "="*50)
    print("测试模型AdaIN功能")
    print("="*50)
    
    # 配置参数
    config = {
        'input_dim': 128,
        'd_model': 256,
        'nhead': 8,
        'num_layers': 2,
        'num_classes': 5,
        'dropout': 0.1,
        'use_adain': True,
        'hidden_dim': 256,
        'bidirectional': True
    }
    
    # 测试数据
    batch_size, seq_len, input_dim = 4, 20, 128
    x = torch.randn(batch_size, seq_len, input_dim)
    style_x = torch.randn(batch_size, seq_len, input_dim)
    
    models = {
        'TransformerModel': TransformerModel(config),
        'LSTMModel': LSTMModel(config),
        'GraphTransformer': GraphTransformer(config)
    }
    
    for model_name, model in models.items():
        print(f"\n测试 {model_name}:")
        
        # 测试不使用AdaIN
        model.eval()
        with torch.no_grad():
            output1 = model(x)
            print(f"  不使用AdaIN输出形状: {output1.shape}")
            
            # 测试使用AdaIN
            output2 = model(x, style_x=style_x)
            print(f"  使用AdaIN输出形状: {output2.shape}")
            
            # 验证输出形状一致
            assert output1.shape == output2.shape, f"{model_name} 输出形状不一致"
            
            # 验证AdaIN确实产生了不同的输出
            diff = torch.abs(output1 - output2).mean()
            print(f"  AdaIN前后输出差异: {diff.item():.6f}")
            
            # AdaIN应该产生不同的输出（除非非常巧合）
            assert diff > 1e-6, f"{model_name} AdaIN没有产生明显的输出差异"
            
        print(f"  ✅ {model_name} AdaIN功能测试通过")


def test_trainer_with_adain():
    """测试训练器的AdaIN功能"""
    print("\n" + "="*50)
    print("测试训练器AdaIN功能")
    print("="*50)
    
    # 创建配置
    config = {
        'input_dim': 64,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'num_classes': 3,
        'dropout': 0.1,
        'use_adain': True,
        'adain_prob': 0.8,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_epochs': 2,  # 只测试2个epoch
        'batch_size': 16,
        'early_stopping_patience': 5
    }
    
    # 创建合成数据
    data_loader = DataLoader(config)
    synthetic_data, synthetic_labels, synthetic_domains = data_loader.generate_synthetic_data(
        num_samples=100,  # 小数据集用于测试
        seq_len=20,
        input_dim=config['input_dim'],
        num_domains=3,
        num_classes=config['num_classes']
    )
    
    # 分割数据
    train_data, val_data, test_data = data_loader.split_data_by_domain(
        synthetic_data, synthetic_labels, synthetic_domains,
        train_domains=[0, 1],
        val_domains=[0, 1],
        test_domains=[2]
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_data, val_data, test_data
    )
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    model = TransformerModel(config)
    
    # 创建训练器
    trainer = Trainer(model, config)
    trainer.model_name = "AdaIN_Test"
    
    print(f"AdaIN配置: 启用={trainer.use_adain}, 概率={trainer.adain_prob}")
    
    # 进行简短的训练测试
    print("\n开始训练测试...")
    try:
        results = trainer.train(train_loader, val_loader)
        print(f"训练完成，最佳验证损失: {results['best_val_loss']:.6f}")
        print("✅ 训练器AdaIN功能测试通过")
    except Exception as e:
        print(f"❌ 训练器AdaIN功能测试失败: {e}")
        raise


def main():
    """主测试函数"""
    print("🧪 开始AdaIN功能测试")
    print("="*80)
    
    try:
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 运行测试
        test_adain_layer()
        test_models_with_adain()
        test_trainer_with_adain()
        
        print("\n" + "="*80)
        print("🎉 所有AdaIN功能测试通过！")
        print("="*80)
        
        # 显示AdaIN功能说明
        print("\n📝 AdaIN功能说明:")
        print("1. AdaIN (Adaptive Instance Normalization) 已成功集成到所有3个模型中")
        print("2. 公式: AdaIN(z1,z2) = mean(z2) + std(z2) * (z1 - mean(z1)) / std(z1)")
        print("3. z1 = encoder(x1) - 原始输入的编码")
        print("4. z2 = encoder(x2) - 从其他域随机采样的输入编码")
        print("5. 训练时会以指定概率使用AdaIN进行域风格转换")
        print("6. 可通过配置文件中的 'use_adain' 和 'adain_prob' 参数控制")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 