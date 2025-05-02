import torch
import torch.nn as nn
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 初始化一个全零的位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 逐个位置计算
        for pos in range(max_len):
            # 逐个维度计算
            for i in range(0, d_model, 2):
                # 计算原始公式中的分母项
                denominator = 10000 ** (i / d_model)

                # 计算正弦项（偶数维度）
                pe[pos, i] = math.sin(pos / denominator)

                # 处理奇数维度（注意边界检查）
                if i + 1 < d_model:
                    denominator = 10000 ** ((i + 1) / d_model)
                    pe[pos, i + 1] = math.cos(pos / denominator)

        # 添加batch维度 -> [1, max_len, d_model]以统一模型接口
        pe = pe.unsqueeze(0)

        # 注册为不需要梯度的持久化张量
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x的形状：[batch_size, seq_len, d_model]
        # 截取与输入序列长度匹配的位置编码
        seq_len = x.size(1)
        position_encoding = self.pe[:, :seq_len]
        # 这里有广播机制在这里，position_encoding 的第一个维度是 1，因此可以扩展为 batch_size，从而与输入张量 x 的形状匹配。

        # 加到输入上
        return x + position_encoding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear transformations
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(output)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        dropout=0.1,
        max_len=5000,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # 将模型移动到指定设备
        self.to(device)

    def forward(self, x, mask=None):
        # 确保输入在正确的设备上
        x = x.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # Embedding and positional encoding
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
