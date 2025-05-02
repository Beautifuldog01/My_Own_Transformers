import torch
import tiktoken
from typing import Tuple


def create_padding_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    创建填充掩码。
    参数：
    - seq: 输入序列，形状为 [batch_size, seq_len]
    返回：
    - 掩码张量，形状为 [batch_size, 1, 1, seq_len]
    """
    mask = (seq != 0).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size: int) -> torch.Tensor:
    """
    创建前瞻掩码。
    参数：
    - size: 序列长度
    返回：
    - 掩码张量，形状为 [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def create_masks(
    src: torch.Tensor, tgt: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建源序列和目标序列的掩码。
    参数：
    - src: 源序列，形状为 [batch_size, src_seq_len]
    - tgt: 目标序列，形状为 [batch_size, tgt_seq_len]
    返回：
    - src_mask: 源序列掩码
    - tgt_mask: 目标序列掩码
    """
    src_mask = create_padding_mask(src)
    tgt_mask = create_padding_mask(tgt) & create_look_ahead_mask(tgt.size(1))
    return src_mask, tgt_mask


def get_tokenizer() -> tiktoken.Encoding:
    """
    获取分词器。
    返回：
    - 分词器实例
    """
    return tiktoken.get_encoding("cl100k_base")


def encode_text(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """
    将文本编码为token。
    参数：
    - text: 输入文本
    - tokenizer: 分词器
    返回：
    - token张量，形状为 [1, seq_len]
    """
    tokens = tokenizer.encode(text)
    return torch.tensor(tokens).unsqueeze(0)
