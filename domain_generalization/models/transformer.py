import torch
import torch.nn as nn
import math
from .base_model import BaseModel, AdaIN


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(BaseModel):
    """
    用于领域泛化的Transformer模型
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 128)
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.num_classes = config.get('num_classes', 5)  # 新增，默认5类
        self.output_dim = self.num_classes  # 保证输出维度等于类别数
        self.dropout = config.get('dropout', 0.1)
        self.use_adain = config.get('use_adain', True)  # 是否使用AdaIN
        
        # 输入投影层
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        
        # AdaIN层
        self.adain = AdaIN(eps=1e-5)
        
        # 输出投影层，最后一层输出1个回归值
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)  # output_dim=1
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
    def encode(self, x):
        """
        编码输入序列
        
        Args:
            x: 输入张量
            
        Returns:
            编码后的特征
        """
        # 检查输入是否有NaN
        if torch.isnan(x).any():
            print("警告：输入数据包含NaN值")
            x = torch.nan_to_num(x, nan=0.0)
        
        # 输入投影
        x = self.input_projection(x)
        
        # 检查投影后是否有NaN
        if torch.isnan(x).any():
            print("警告：输入投影后包含NaN值")
            x = torch.nan_to_num(x, nan=0.0)
        
        # 添加位置编码
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # 检查位置编码后是否有NaN
        if torch.isnan(x).any():
            print("警告：位置编码后包含NaN值")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Transformer编码
        encoded = self.transformer_encoder(x)
        
        # 检查编码后是否有NaN
        if torch.isnan(encoded).any():
            print("警告：Transformer编码后包含NaN值")
            encoded = torch.nan_to_num(encoded, nan=0.0)
        
        return encoded
    
    def forward(self, x, style_x=None, **kwargs):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            style_x: 用于AdaIN的样式输入张量 [batch_size, seq_len, input_dim]，如果为None则不使用AdaIN
            
        Returns:
            模型输出 [batch_size, 1]
        """
        # 编码原始输入
        encoded = self.encode(x)  # [batch_size, seq_len, d_model]
        
        # 如果提供了样式输入且启用AdaIN，则应用AdaIN
        if style_x is not None and self.use_adain:
            style_encoded = self.encode(style_x)  # [batch_size, seq_len, d_model]
            encoded = self.adain(encoded, style_encoded)
        
        # 使用稳定的池化操作
        pooled = encoded.mean(dim=1)  # [batch_size, d_model]
        
        # 检查池化后是否有NaN
        if torch.isnan(pooled).any():
            print("警告：池化后包含NaN值")
            pooled = torch.nan_to_num(pooled, nan=0.0)
        
        output = self.output_projection(pooled)  # [batch_size, 1]
        
        # 检查最终输出是否有NaN
        if torch.isnan(output).any():
            print("警告：最终输出包含NaN值")
            output = torch.nan_to_num(output, nan=0.0)
        
        return output
    
    def get_shared_features(self, x, **kwargs):
        """
        获取共享特征
        
        Args:
            x: 输入张量
            
        Returns:
            共享特征
        """
        encoded = self.encode(x)
        return torch.mean(encoded, dim=1) 