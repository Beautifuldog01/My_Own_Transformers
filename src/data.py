import os
import torch
from typing import List, Tuple
from utils import encode_text, get_tokenizer


class TranslationDataset(torch.utils.data.Dataset):
    """
    翻译数据集类。
    参数：
    - data: 数据列表，每个元素是(源文本, 目标文本)的元组
    - tokenizer: 分词器
    - max_seq_len: 最大序列长度
    """

    def __init__(
        self,
        data: List[Tuple[str, str]],
        tokenizer,
        max_seq_len: int = 512,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        src_tokens = encode_text(src_text, self.tokenizer)
        tgt_tokens = encode_text(tgt_text, self.tokenizer)

        # 截断过长的序列
        if src_tokens.size(1) > self.max_seq_len:
            src_tokens = src_tokens[:, : self.max_seq_len]
        if tgt_tokens.size(1) > self.max_seq_len:
            tgt_tokens = tgt_tokens[:, : self.max_seq_len]

        return src_tokens, tgt_tokens


def pad_sequence(sequences, padding_value=0):
    """
    填充序列到相同长度。
    参数：
    - sequences: 序列列表
    - padding_value: 填充值
    返回：
    - 填充后的张量
    """
    max_len = max(seq.size(1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        if seq.size(1) < max_len:
            padding = torch.zeros(1, max_len - seq.size(1), dtype=seq.dtype)
            padded_seq = torch.cat([seq, padding], dim=1)
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    return torch.cat(padded_sequences, dim=0)


def create_dataloader(
    data: List[Tuple[str, str]],
    batch_size: int,
    max_seq_len: int = 512,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    创建数据加载器。
    参数：
    - data: 数据列表，每个元素是(源文本, 目标文本)的元组
    - batch_size: 批次大小
    - max_seq_len: 最大序列长度
    - shuffle: 是否打乱数据
    返回：
    - 数据加载器
    """
    tokenizer = get_tokenizer()
    dataset = TranslationDataset(data, tokenizer, max_seq_len)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: (
            pad_sequence([item[0] for item in x]),
            pad_sequence([item[1] for item in x]),
        ),
    )
    return dataloader


def load_data_from_file(
    file_path: str,
    batch_size: int,
    max_seq_len: int = 512,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    """
    从文件加载数据并创建数据加载器。
    参数：
    - file_path: 文件路径
    - batch_size: 批次大小
    - max_seq_len: 最大序列长度
    - shuffle: 是否打乱数据
    返回：
    - 数据加载器
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                src, tgt = line.split("\t")
                data.append((src, tgt))
            except ValueError:
                print(f"跳过无效行: {line}")

    return create_dataloader(data, batch_size, max_seq_len, shuffle)
