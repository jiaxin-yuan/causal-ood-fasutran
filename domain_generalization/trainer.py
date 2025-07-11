import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import json
import os
from typing import Dict, List, Optional
from tqdm import tqdm
import logging
from datetime import datetime
import random


class Trainer:
    """用于领域泛化模型的训练器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.model_name = 'model'  # 默认名称，可以后续设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # AdaIN相关配置
        self.use_adain = config.get('use_adain', True)
        self.adain_prob = config.get('adain_prob', 0.5)  # 使用AdaIN的概率
        
        # 存储所有训练数据用于域随机采样
        self.domain_data_cache = {}  # {domain_id: [samples]}
        
        # 默认logger，可以后续被替换
        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
        # 训练参数
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.num_epochs = config.get('num_epochs', 100)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        
        # 优化器和调度器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # 损失函数 - 使用Huber Loss，对异常值更稳定
        self.criterion = nn.SmoothL1Loss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        # 添加模型参数检查
        self._check_model_parameters()
        
        self.logger.info(f"训练器初始化完成，设备: {self.device}")
        self.logger.info(f"学习率: {self.learning_rate}, 权重衰减: {self.weight_decay}")
        self.logger.info(f"训练轮数: {self.num_epochs}, 早停耐心: {self.early_stopping_patience}")
        self.logger.info(f"AdaIN配置: 启用={self.use_adain}, 概率={self.adain_prob}")
        
    def _setup_logger(self):
        """设置日志记录器"""
        # 创建日志文件名（包含时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{self.model_name}_{timestamp}.log"
        log_filepath = os.path.join(self.log_dir, log_filename)
        
        # 创建专用的logger
        self.logger = logging.getLogger(f"trainer_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        
        # 清除之前的handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
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
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 防止日志重复
        self.logger.propagate = False
        
        self.logger.info(f"日志文件已创建: {log_filepath}")
        
    def _check_model_parameters(self):
        """检查模型参数是否有问题"""
        self.logger.info("检查模型参数...")
        total_params = 0
        nan_params = 0
        inf_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if torch.isnan(param).any():
                nan_params += torch.isnan(param).sum().item()
                self.logger.warning(f"参数 {name} 包含 {torch.isnan(param).sum().item()} 个NaN值")
            if torch.isinf(param).any():
                inf_params += torch.isinf(param).sum().item()
                self.logger.warning(f"参数 {name} 包含 {torch.isinf(param).sum().item()} 个无穷值")
        
        self.logger.info(f"总参数数量: {total_params:,}")
        self.logger.info(f"NaN参数数量: {nan_params}")
        self.logger.info(f"无穷参数数量: {inf_params}")
        
        if nan_params > 0 or inf_params > 0:
            self.logger.warning("模型参数包含异常值，重新初始化...")
            self._reinitialize_model()
    
    def _reinitialize_model(self):
        """重新初始化模型参数"""
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
        
        self.logger.info("模型参数重新初始化完成")
        
    def _build_domain_cache(self, dataloader):
        """构建域数据缓存，用于随机采样"""
        if not self.use_adain:
            return
            
        self.logger.info("构建域数据缓存...")
        self.domain_data_cache = {}
        
        for batch in dataloader:
            data = batch['data'].float()
            domains = batch['domain']
            
            for i in range(len(data)):
                domain_id = domains[i].item()
                if domain_id not in self.domain_data_cache:
                    self.domain_data_cache[domain_id] = []
                self.domain_data_cache[domain_id].append(data[i])
        
        # 转换为tensor
        for domain_id in self.domain_data_cache:
            self.domain_data_cache[domain_id] = torch.stack(self.domain_data_cache[domain_id])
            
        self.logger.info(f"域数据缓存构建完成，包含 {len(self.domain_data_cache)} 个域")
        for domain_id, samples in self.domain_data_cache.items():
            self.logger.info(f"  域 {domain_id}: {len(samples)} 个样本")
    
    def _get_random_domain_samples(self, batch_size, current_domains=None):
        """从所有域中随机采样样本作为style输入"""
        if not self.use_adain or not self.domain_data_cache:
            return None
            
        style_samples = []
        available_domains = list(self.domain_data_cache.keys())
        
        for i in range(batch_size):
            # 如果提供了当前域，尝试从不同域采样
            if current_domains is not None and len(available_domains) > 1:
                current_domain = current_domains[i].item()
                other_domains = [d for d in available_domains if d != current_domain]
                if other_domains:
                    domain_id = random.choice(other_domains)
                else:
                    domain_id = random.choice(available_domains)
            else:
                domain_id = random.choice(available_domains)
            
            # 从选定域中随机采样一个样本
            domain_samples = self.domain_data_cache[domain_id]
            sample_idx = random.randint(0, len(domain_samples) - 1)
            style_samples.append(domain_samples[sample_idx])
        
        return torch.stack(style_samples)
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        total = 0
        nan_batches = 0
        
        # 使用tqdm显示进度条
        pbar = tqdm(train_loader, desc=f"训练{self.model_name}", leave=False)
        
        for batch_idx, batch in enumerate(pbar):
            data = batch['data'].float().to(self.device)
            labels = batch['rtime'].float().to(self.device)
            domains = batch['domain'] if 'domain' in batch else None
            
            # 检查输入数据
            if torch.isnan(data).any() or torch.isnan(labels).any():
                self.logger.warning(f"批次 {batch_idx} 输入数据包含NaN值，跳过")
                continue
            
            if torch.isinf(data).any() or torch.isinf(labels).any():
                self.logger.warning(f"批次 {batch_idx} 输入数据包含无穷值，跳过")
                continue
            
            # 获取随机域样本作为style输入
            style_x = None
            if self.use_adain and random.random() < self.adain_prob:
                style_x = self._get_random_domain_samples(data.size(0), domains)
                if style_x is not None:
                    style_x = style_x.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            if style_x is not None:
                outputs = self.model(data, style_x=style_x).squeeze(-1)
            else:
                outputs = self.model(data).squeeze(-1)
            
            # 检查模型输出
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                self.logger.warning(f"批次 {batch_idx} 模型输出包含异常值")
                self.logger.warning(f"outputs范围: {outputs.min().item():.4f} - {outputs.max().item():.4f}")
                self.logger.warning(f"labels范围: {labels.min().item():.4f} - {labels.max().item():.4f}")
                
                # 检查模型参数是否有异常
                param_nan = False
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any():
                        self.logger.warning(f"参数 {name} 包含NaN值")
                        param_nan = True
                        break
                
                if param_nan:
                    self.logger.warning("检测到模型参数异常，重新初始化...")
                    self._reinitialize_model()
                    # 重新创建优化器
                    self.optimizer = optim.Adam(
                        self.model.parameters(), 
                        lr=self.learning_rate, 
                        weight_decay=self.weight_decay
                    )
                
                nan_batches += 1
                continue  # 跳过这个batch
            
            # 计算损失
            loss = self.criterion(outputs, labels)
            
            # 检查损失是否为nan
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"批次 {batch_idx} 损失为异常值: {loss.item()}")
                nan_batches += 1
                continue  # 跳过这个batch
            
            # 反向传播
            loss.backward()
            
            # 检查梯度是否有异常
            grad_nan = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        self.logger.warning(f"参数 {name} 的梯度包含异常值")
                        grad_nan = True
                        break
            
            if grad_nan:
                self.logger.warning("检测到梯度异常，跳过这个批次")
                self.optimizer.zero_grad()
                continue
            
            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计信息
            total_loss += loss.item() * labels.size(0)
            total_mae += torch.abs(outputs - labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            if batch_idx % 10 == 0:  # 每10个batch更新一次
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MAE': f'{torch.abs(outputs - labels).mean().item():.4f}'
                })
                
                # 记录详细的批次信息到日志
                if batch_idx % 50 == 0:  # 每50个batch记录一次详细信息
                    self.logger.debug(f"批次 {batch_idx}: Loss={loss.item():.6f}, MAE={torch.abs(outputs - labels).mean().item():.6f}")
        
        if total == 0:
            self.logger.error("所有批次都被跳过，无法计算epoch损失")
            return float('inf'), float('inf')
        
        epoch_loss = total_loss / total
        epoch_mae = total_mae / total
        
        if nan_batches > 0:
            self.logger.warning(f"本epoch跳过了 {nan_batches} 个异常批次")
        
        return epoch_loss, epoch_mae
    
    def validate_epoch(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"验证{self.model_name}", leave=False)
            for batch in pbar:
                data = batch['data'].float().to(self.device)
                labels = batch['rtime'].float().to(self.device)
                domains = batch['domain'] if 'domain' in batch else None
                
                # 检查输入数据
                if torch.isnan(data).any() or torch.isnan(labels).any():
                    continue
                
                if torch.isinf(data).any() or torch.isinf(labels).any():
                    continue
                
                # 在验证时也可以使用AdaIN（以较低概率）
                style_x = None
                if self.use_adain and random.random() < self.adain_prob * 0.5:  # 验证时使用较低概率
                    style_x = self._get_random_domain_samples(data.size(0), domains)
                    if style_x is not None:
                        style_x = style_x.to(self.device)
                
                # 前向传播
                if style_x is not None:
                    outputs = self.model(data, style_x=style_x).squeeze(-1)
                else:
                    outputs = self.model(data).squeeze(-1)
                
                # 检查输出
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue
                
                # 计算损失
                loss = self.criterion(outputs, labels)
                
                # 检查损失
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # 统计信息
                total_loss += loss.item() * labels.size(0)
                total_mae += torch.abs(outputs - labels).sum().item()
                total += labels.size(0)
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'MAE': f'{torch.abs(outputs - labels).mean().item():.4f}'
                })
        
        if total == 0:
            return float('inf'), float('inf')
        
        epoch_loss = total_loss / total
        epoch_mae = total_mae / total
        
        return epoch_loss, epoch_mae
    
    def train(self, train_loader, val_loader, save_path=None):
        """训练模型"""
        best_val_loss = float('inf')
        best_val_mae = float('inf')
        patience_counter = 0
        
        self.logger.info(f"开始训练模型: {self.model_name}")
        self.logger.info(f"训练设备: {self.device}")
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"训练集大小: {len(train_loader.dataset)}")
        self.logger.info(f"验证集大小: {len(val_loader.dataset)}")
        self.logger.info(f"批次大小: {train_loader.batch_size}")
        self.logger.info(f"学习率: {self.learning_rate}")
        self.logger.info(f"权重衰减: {self.weight_decay}")
        
        # 构建域数据缓存用于AdaIN
        self._build_domain_cache(train_loader)
        
        print(f"\n" + "="*80)
        print(f"开始训练模型: {self.model_name}")
        print("="*80)
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            
            print(f"\n📊 Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_mae = self.train_epoch(train_loader)
            
            # 如果训练失败，提前结束
            if train_loss == float('inf'):
                self.logger.error("训练失败，提前结束")
                break
            
            # 验证
            val_loss, val_mae = self.validate_epoch(val_loader)
            
            # 更新学习率
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            # 存储历史记录
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start_time
            
            # 记录epoch结果到日志
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} 完成:")
            self.logger.info(f"  训练损失(MSE): {train_loss:.6f}")
            self.logger.info(f"  训练MAE: {train_mae:.6f}")
            self.logger.info(f"  验证损失(MSE): {val_loss:.6f}")
            self.logger.info(f"  验证MAE: {val_mae:.6f}")
            self.logger.info(f"  学习率: {new_lr:.8f}")
            self.logger.info(f"  Epoch时间: {epoch_time:.2f}秒")
            
            # 打印详细的训练信息到控制台
            print(f"🚀 训练结果:")
            print(f"   训练损失(MSE): {train_loss:.6f}")
            print(f"   训练MAE:      {train_mae:.6f}")
            print(f"🎯 验证结果:")
            print(f"   验证损失(MSE): {val_loss:.6f}")
            print(f"   验证MAE:      {val_mae:.6f}")
            print(f"⚙️  训练参数:")
            print(f"   学习率:       {new_lr:.8f}")
            if old_lr != new_lr:
                print(f"   学习率已调整: {old_lr:.8f} → {new_lr:.8f}")
                self.logger.info(f"学习率已调整: {old_lr:.8f} → {new_lr:.8f}")
            print(f"   Epoch时间:    {epoch_time:.2f}秒")
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_mae = val_mae
                patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    self.save_model(save_path)
                    print(f"💾 最佳模型已保存到: {save_path}")
                    self.logger.info(f"最佳模型已保存到: {save_path}")
                    
                print(f"🎉 新的最佳验证损失: {best_val_loss:.6f}")
                self.logger.info(f"新的最佳验证损失: {best_val_loss:.6f}")
            else:
                patience_counter += 1
                print(f"⏰ 早停计数器: {patience_counter}/{self.early_stopping_patience}")
                self.logger.info(f"早停计数器: {patience_counter}/{self.early_stopping_patience}")
                
            if patience_counter >= self.early_stopping_patience:
                print(f"\n⏹️  在第 {epoch+1} 个epoch后触发早停")
                self.logger.info(f"在第 {epoch+1} 个epoch后触发早停")
                break
        
        print(f"\n" + "="*80)
        print(f"模型 {self.model_name} 训练完成")
        print(f"🏆 最佳验证损失: {best_val_loss:.6f}")
        print(f"🏆 最佳验证MAE:  {best_val_mae:.6f}")
        print("="*80)
        
        # 记录最终结果
        self.logger.info(f"模型 {self.model_name} 训练完成")
        self.logger.info(f"最佳验证损失: {best_val_loss:.6f}")
        self.logger.info(f"最佳验证MAE: {best_val_mae:.6f}")
        self.logger.info(f"训练总共进行了 {len(self.train_losses)} 个epoch")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss,
            'best_val_mae': best_val_mae
        }
    
    def save_model(self, path):
        """保存模型检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
        }, path)
        self.logger.info(f"模型检查点已保存到: {path}")
    
    def load_model(self, path):
        """加载模型检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复训练历史
        history = checkpoint.get('train_history', {})
        self.train_losses = history.get('train_losses', [])
        self.val_losses = history.get('val_losses', [])
        
        self.logger.info(f"模型检查点已从 {path} 加载")
        
        return checkpoint.get('config', {})