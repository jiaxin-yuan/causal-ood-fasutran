import torch
import torch.nn as nn
from .base_model import BaseModel, AdaIN


class LSTMModel(BaseModel):
    """
    用于领域泛化的LSTM模型
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_classes = config.get('num_classes', 5)  # 新增，默认5类
        self.output_dim = self.num_classes  # 保证输出维度等于类别数
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.bidirectional = config.get('bidirectional', True)
        self.use_adain = config.get('use_adain', True)  # 是否使用AdaIN
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        # AdaIN层
        lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        self.adain = AdaIN(eps=1e-5)
        
        # 输出层，输出1个回归值
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(lstm_output_dim // 2, 1)
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
            elif isinstance(module, nn.LSTM):
                # LSTM权重初始化
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
        
    def encode(self, x):
        """
        编码输入序列
        
        Args:
            x: 输入张量
            
        Returns:
            LSTM输出
        """
        # 检查输入是否有NaN
        if torch.isnan(x).any():
            print("警告：LSTM输入数据包含NaN值")
            x = torch.nan_to_num(x, nan=0.0)
        
        lstm_output, (hidden, cell) = self.lstm(x)
        
        # 检查LSTM输出是否有NaN
        if torch.isnan(lstm_output).any():
            print("警告：LSTM输出包含NaN值")
            lstm_output = torch.nan_to_num(lstm_output, nan=0.0)
        
        return lstm_output
    
    def forward(self, x, style_x=None, **kwargs):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_dim]
            style_x: 用于AdaIN的样式输入张量 [batch_size, seq_len, input_dim]，如果为None则不使用AdaIN
            
        Returns:
            模型输出 [batch_size, 1]
        """
        # 检查输入是否有NaN
        if torch.isnan(x).any():
            print("警告：LSTM前向传播输入包含NaN值")
            x = torch.nan_to_num(x, nan=0.0)
        
        # 编码原始输入
        out, _ = self.lstm(x)
        
        # 如果提供了样式输入且启用AdaIN，则应用AdaIN
        if style_x is not None and self.use_adain:
            style_out, _ = self.lstm(style_x)
            out = self.adain(out, style_out)
        
        # 检查LSTM输出是否有NaN
        if torch.isnan(out).any():
            print("警告：LSTM前向传播输出包含NaN值")
            out = torch.nan_to_num(out, nan=0.0)
        
        out = out[:, -1, :]  # 取最后一个时间步
        
        # 检查最后时间步输出是否有NaN
        if torch.isnan(out).any():
            print("警告：LSTM最后时间步输出包含NaN值")
            out = torch.nan_to_num(out, nan=0.0)
        
        out = self.output_layer(out)  # [batch_size, 1]
        
        # 检查最终输出是否有NaN
        if torch.isnan(out).any():
            print("警告：LSTM最终输出包含NaN值")
            out = torch.nan_to_num(out, nan=0.0)
        
        return out
    
    def get_shared_features(self, x, **kwargs):
        """
        获取共享特征
        
        Args:
            x: 输入张量
            
        Returns:
            共享特征
        """
        encoded = self.encode(x)
        # 对于双向LSTM，使用前向和后向隐藏状态
        if self.bidirectional:
            # 连接最后一个前向隐藏状态和第一个后向隐藏状态
            forward_hidden = encoded[:, -1, :self.hidden_dim]
            backward_hidden = encoded[:, 0, self.hidden_dim:]
            shared_features = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            shared_features = encoded[:, -1, :]  # 最后一个隐藏状态
        
        return shared_features 