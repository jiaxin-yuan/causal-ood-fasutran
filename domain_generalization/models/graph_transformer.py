import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel, AdaIN


class GraphAttentionLayer(nn.Module):
    """
    图注意力层
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 权重矩阵
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        
        # 注意力参数
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        """
        前向传播
        
        Args:
            input: 输入节点特征
            adj: 邻接矩阵
            
        Returns:
            更新后的节点特征
        """
        # 检查输入是否有NaN
        if torch.isnan(input).any():
            print("警告：GAT输入包含NaN值")
            input = torch.nan_to_num(input, nan=0.0)
        
        if torch.isnan(adj).any():
            print("警告：邻接矩阵包含NaN值")
            adj = torch.nan_to_num(adj, nan=0.0)
        
        Wh = torch.mm(input, self.W)
        
        # 检查权重变换后是否有NaN
        if torch.isnan(Wh).any():
            print("警告：GAT权重变换后包含NaN值")
            Wh = torch.nan_to_num(Wh, nan=0.0)
        
        a_input = self._prepare_attentional_mechanism_input(Wh)
        
        # 检查注意力输入是否有NaN
        if torch.isnan(a_input).any():
            print("警告：GAT注意力输入包含NaN值")
            a_input = torch.nan_to_num(a_input, nan=0.0)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # 检查注意力分数是否有NaN
        if torch.isnan(e).any():
            print("警告：GAT注意力分数包含NaN值")
            e = torch.nan_to_num(e, nan=0.0)

        # 计算注意力权重 - 添加数值稳定性
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # 添加数值稳定性：在softmax之前裁剪极值
        attention = torch.clamp(attention, min=-1e10, max=1e10)
        attention = F.softmax(attention, dim=1)
        
        # 检查注意力权重是否有NaN
        if torch.isnan(attention).any():
            print("警告：GAT注意力权重包含NaN值")
            attention = torch.nan_to_num(attention, nan=0.0)
            # 重新归一化
            attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-8)
        
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        
        # 检查最终输出是否有NaN
        if torch.isnan(h_prime).any():
            print("警告：GAT最终输出包含NaN值")
            h_prime = torch.nan_to_num(h_prime, nan=0.0)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        """
        准备注意力机制的输入
        
        Args:
            Wh: 变换后的节点特征
            
        Returns:
            注意力机制的输入矩阵
        """
        N = Wh.size()[0]
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class GraphTransformer(BaseModel):
    """
    Graph Transformer model for DG
    """
    def __init__(self, config):
        super().__init__(config)
        
        self.input_dim = config.get('input_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_classes = config.get('num_classes', 5)  # added, 5 by default
        self.output_dim = self.num_classes  # ensure output dimension = number of categories
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.use_adain = config.get('use_adain', True)  # whether to use AdaIN
        
        # Graph attention
        self.gat_layers = nn.ModuleList()
        # the first layer: in_features=input_dim, out_features=hidden_dim
        self.gat_layers.append(GraphAttentionLayer(
            self.input_dim, self.hidden_dim, 
            dropout=self.dropout, concat=True
        ))
        # following layer: in_features=hidden_dim, out_features=hidden_dim
        for _ in range(self.num_layers - 1):
            self.gat_layers.append(GraphAttentionLayer(
                self.hidden_dim, self.hidden_dim,
                dropout=self.dropout, concat=True
            ))
        
        # AdaIN layer
        self.adain = AdaIN(eps=1e-5)
        
        # output layer，output 1 regression value
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        # initialise weights
        self._init_weights()
        
    def _init_weights(self):
        """initialise model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # use Xavier to initialise
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
    def encode(self, x, adj=None):
        """
        encode input sequence
        
        Args:
            x: input tensor
            adj: adjacency matrix，if it is None then create a fully connected graph
            
        Returns:
            features after encoding
        """
        # check whether input has NaN
        if torch.isnan(x).any():
            print("warning：GraphTransformer inputs include NaN value")
            x = torch.nan_to_num(x, nan=0.0)
        
        # if there's no adjacency matrix provided, then create a fully connected graph
        if adj is None:
            batch_size, seq_len, _ = x.shape
            adj = torch.ones(batch_size, seq_len, seq_len).to(x.device)
        
        # deal with samples in a batch
        encoded_outputs = []
        for i in range(x.shape[0]):
            node_features = x[i]  # (seq_len, input_dim)
            adjacency = adj[i]    # (seq_len, seq_len)
            
            # check whether there's NaN in the batch samples
            if torch.isnan(node_features).any():
                print(f"warning：NaN in features of sample{i}")
                node_features = torch.nan_to_num(node_features, nan=0.0)
            
            # apply GAT layer
            hidden = node_features
            for gat_layer in self.gat_layers:
                hidden = gat_layer(hidden, adjacency)
                
                # check whether output of each layer has NaN
                if torch.isnan(hidden).any():
                    print(f"warning：NaN in output of sample{i}")
                    hidden = torch.nan_to_num(hidden, nan=0.0)
            
            encoded_outputs.append(hidden)
        
        # stack outputs
        encoded = torch.stack(encoded_outputs, dim=0)
        
        # check whether NaN in encoded output
        if torch.isnan(encoded).any():
            print("warning：NaN in GraphTransformer encoding output")
            encoded = torch.nan_to_num(encoded, nan=0.0)
        
        return encoded
    
    def forward(self, x, style_x=None, **kwargs):
        """
        forward
        
        Args:
            x: input tensor [batch_size, seq_len, input_dim]
            style_x: another input tensor for AdaIN [batch_size, seq_len, input_dim]，None if not use AdaIN
            
        Returns:
            model output [batch_size, 1]
        """
        # encoding original input
        h = self.encode(x)
        
        # if style_x provided and use_AdaIN is True，then apply AdaIN
        if style_x is not None and self.use_adain:
            style_h = self.encode(style_x)
            h = self.adain(h, style_h)
        
        # check whether NaN in encoded output 
        if torch.isnan(h).any():
            print("warning：NaN in GraphTransformer forward output") 
            h = torch.nan_to_num(h, nan=0.0)
        
        h = h.mean(dim=1)  # pooling
        
        # check whether NaN exists after pooling 
        if torch.isnan(h).any():
            print("warning：NaN in GraphTransformer after pooling")
            h = torch.nan_to_num(h, nan=0.0)
        
        out = self.output_layer(h)  # [batch_size, 1]
        
        # check whether NaN in final output 
        if torch.isnan(out).any():
            print("warning：NaN in GraphTransformer final output")
            out = torch.nan_to_num(out, nan=0.0)
        
        return out
    
    def get_shared_features(self, x, adj=None, **kwargs):
        """
        get shared features
        
        Args:
            x: input tensor
            adj: adjacency matrix
            
        Returns:
            shared features
        """
        encoded = self.encode(x, adj, **kwargs)
        return torch.mean(encoded, dim=1) 