import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码模块。
        参数：
        - d_model: 特征维度（embedding 的维度）。
        - max_len: 最大序列长度。
        """
        super().__init__()

        # 初始化一个全零的位置编码矩阵，形状为 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # 逐个位置计算位置编码
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                # 计算公式中的分母项
                denominator = 10000 ** (i / d_model)

                # 偶数维度使用正弦函数
                pe[pos, i] = math.sin(pos / denominator)

                # 奇数维度使用余弦函数（注意边界检查）
                if i + 1 < d_model:
                    denominator = 10000 ** ((i + 1) / d_model)
                    pe[pos, i + 1] = math.cos(pos / denominator)

        # 添加 batch 维度，形状变为 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为持久化张量，不需要梯度
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        前向传播函数。
        参数：
        - x: 输入张量，形状为 [batch_size, seq_len, d_model]。
        返回：
        - 加入位置编码后的张量，形状与输入相同。
        """
        # 获取输入序列的长度
        seq_len = x.size(1)

        # 截取与输入序列长度匹配的位置编码
        position_encoding = self.pe[:, :seq_len]

        # 利用广播机制，将位置编码加到输入张量上
        return x + position_encoding


# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        初始化多头注意力模块。
        参数：
        - d_model: 输入特征的维度。
        - num_heads: 注意力头的数量。
        - dropout: Dropout 概率。
        """
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
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k]))

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数。
        参数：
        - q: 查询张量，形状为 [batch_size, seq_len, d_model]
        - k: 键张量，形状为 [batch_size, seq_len, d_model]
        - v: 值张量，形状为 [batch_size, seq_len, d_model]
        - mask: 掩码张量，形状为 [batch_size, 1, seq_len, seq_len] 或 [batch_size, 1, 1, seq_len]
        返回：
        - 输出张量，形状为 [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)
        device = q.device

        # 线性变换并分头 [batch_size, num_heads, seq_len, d_k]
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数 [batch_size, num_heads, q_len, k_len]
        scale = self.scale.to(device)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # 应用掩码
        if mask is not None:
            # 确保掩码在正确的设备上
            mask = mask.to(device)

            # 处理掩码维度 - 确保掩码维度为4
            if mask.dim() > 4:  # 如果掩码维度大于4，移除多余维度
                mask = mask.squeeze(2)

            # 扩展掩码到所有头
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.expand(-1, self.num_heads, -1, -1)

            # 将掩码应用到分数上，这里其实就是负无穷过softmax了
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重 [batch_size, num_heads, q_len, k_len]
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 计算输出 [batch_size, num_heads, q_len, d_k]
        output = torch.matmul(attn, v)

        # 重新排列维度并合并头 [batch_size, q_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)

        return output


# 前馈网络模块
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈网络模块。
        参数：
        - d_model: 输入特征的维度。
        - d_ff: 前馈网络的隐藏层维度。
        - dropout: Dropout 概率。
        """
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 第一层线性变换
        self.w_2 = nn.Linear(d_ff, d_model)  # 第二层线性变换
        self.dropout = nn.Dropout(dropout)  # Dropout 层
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, x):
        """
        前向传播函数。
        参数：
        - x: 输入张量，形状为 [batch_size, seq_len, d_model]。
        返回：
        - 输出张量，形状为 [batch_size, seq_len, d_model]。
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


# 编码器层模块
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化编码器层模块。
        参数：
        - d_model: 输入特征的维度。
        - num_heads: 注意力头的数量。
        - d_ff: 前馈网络的隐藏层维度。
        - dropout: Dropout 概率。
        """
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)  # 多头自注意力
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 第一层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout 层

    def forward(self, x, mask=None):
        """
        前向传播函数。
        参数：
        - x: 输入张量，形状为 [batch_size, seq_len, d_model]。
        - mask: 掩码张量。
        返回：
        - 输出张量，形状为 [batch_size, seq_len, d_model]。
        """
        # 自注意力层
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接 + 归一化

        # 前馈网络层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # 残差连接 + 归一化

        return x


# 编码器模块
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
        """
        初始化编码器模块。
        参数：
        - vocab_size: 词汇表大小。
        - d_model: 输入特征的维度。
        - num_heads: 注意力头的数量。
        - num_layers: 编码器层的数量。
        - d_ff: 前馈网络的隐藏层维度。
        - dropout: Dropout 概率。
        - max_len: 最大序列长度。
        - device: 设备（如 "cpu" 或 "cuda"）。
        """
        super().__init__()
        self.device = device  # 设备
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.pos_encoding = PositionalEncoding(d_model, max_len)  # 位置编码
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )  # 多个编码器层
        self.norm = nn.LayerNorm(d_model)  # 最后一层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout 层

        # 将模型移动到指定设备
        self.to(device)

    def forward(self, x, mask=None):
        """
        前向传播函数。
        参数：
        - x: 输入张量，形状为 [batch_size, seq_len]。
        - mask: 掩码张量。
        返回：
        - 输出张量，形状为 [batch_size, seq_len, d_model]。
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        # 词嵌入和位置编码
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 通过N即num_layers个编码器层，原文设定是6层
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# 解码器层模块
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化解码器层模块。
        参数：
        - d_model: 输入特征的维度。
        - num_heads: 注意力头的数量。
        - d_ff: 前馈网络的隐藏层维度。
        - dropout: Dropout 概率。
        """
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 带掩码的多头自注意力
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 编码器-解码器注意力
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 第一层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二层归一化
        self.norm3 = nn.LayerNorm(d_model)  # 第三层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout 层

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        前向传播函数。
        参数：
        - x: 解码器输入张量，形状为 [batch_size, seq_len, d_model]。
        - enc_output: 编码器输出张量，形状为 [batch_size, seq_len, d_model]。
        - src_mask: 源序列掩码。
        - tgt_mask: 目标序列掩码。
        返回：
        - 输出张量，形状为 [batch_size, seq_len, d_model]。
        """
        # 带掩码的自注意力层
        attn_output = self.masked_self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))  # 残差连接 + 归一化

        # 编码器-解码器注意力层
        # 这里直接使用src_mask，不需要特殊处理
        # src_mask形状应为 [batch_size, 1, 1, src_len]
        attn_output = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))  # 残差连接 + 归一化

        # 前馈网络层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # 残差连接 + 归一化

        return x


# 解码器模块
class Decoder(nn.Module):
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
        """
        初始化解码器模块。
        参数：
        - vocab_size: 词汇表大小。
        - d_model: 输入特征的维度。
        - num_heads: 注意力头的数量。
        - num_layers: 解码器层的数量。
        - d_ff: 前馈网络的隐藏层维度。
        - dropout: Dropout 概率。
        - max_len: 最大序列长度。
        - device: 设备（如 "cpu" 或 "cuda"）。
        """
        super().__init__()
        self.device = device  # 设备
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词嵌入层
        self.pos_encoding = PositionalEncoding(d_model, max_len)  # 位置编码
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )  # 多个解码器层
        self.norm = nn.LayerNorm(d_model)  # 最后一层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout 层
        self.linear = nn.Linear(d_model, vocab_size)  # 输出线性层

        # 将模型移动到指定设备
        self.to(device)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        前向传播函数。
        参数：
        - x: 解码器输入张量，形状为 [batch_size, seq_len]。
        - enc_output: 编码器输出张量，形状为 [batch_size, seq_len, d_model]。
        - src_mask: 源序列掩码。
        - tgt_mask: 目标序列掩码。
        返回：
        - 输出张量，形状为 [batch_size, seq_len, vocab_size]。
        """
        # 确保输入在正确的设备上
        x = x.to(self.device)
        enc_output = enc_output.to(self.device)
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.to(self.device)

        # 词嵌入和位置编码
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 通过N个解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        # 最后一层归一化
        x = self.norm(x)

        # 线性变换到词汇表大小
        return self.linear(x)
