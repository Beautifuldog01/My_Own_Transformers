import torch
import tiktoken
from models import Encoder, Decoder
from config import Config


def create_padding_mask(seq):
    """
    创建填充掩码。
    参数：
    - seq: 输入序列，形状为 [batch_size, seq_len]
    返回：
    - 掩码张量，形状为 [batch_size, 1, 1, seq_len]
    """
    mask = (seq != 0).unsqueeze(1).unsqueeze(2)
    return mask


def create_look_ahead_mask(size):
    """
    创建前瞻掩码。
    参数：
    - size: 序列长度
    返回：
    - 掩码张量，形状为 [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0


def main():
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # 示例源文本和目标文本
    src_text = "Learning is the best reward."
    tgt_text = "学习是旅途的意义。"

    # 使用分词器编码文本
    src_tokens = tokenizer.encode(src_text)
    tgt_tokens = tokenizer.encode(tgt_text)

    # 转换为张量并添加batch维度
    src_tokens = torch.tensor(src_tokens).unsqueeze(0)
    tgt_tokens = torch.tensor(tgt_tokens).unsqueeze(0)

    # 创建掩码
    src_mask = create_padding_mask(src_tokens)
    tgt_mask = create_padding_mask(tgt_tokens) & create_look_ahead_mask(
        tgt_tokens.size(1)
    )

    # 创建编码器
    encoder = Encoder(
        vocab_size=tokenizer.n_vocab,
        d_model=Config.d_model,
        num_heads=Config.num_heads,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        device=Config.device,
    )

    # 创建解码器
    decoder = Decoder(
        vocab_size=tokenizer.n_vocab,
        d_model=Config.d_model,
        num_heads=Config.num_heads,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
        device=Config.device,
    )

    # 编码器前向传播
    enc_output = encoder(src_tokens, src_mask)

    # 解码器前向传播
    dec_output = decoder(tgt_tokens, enc_output, src_mask, tgt_mask)

    print(f"源文本: {src_text}")
    print(f"目标文本: {tgt_text}")
    print(f"编码器输出形状: {enc_output.shape}")
    print(f"解码器输出形状: {dec_output.shape}")
    print(f"解码器输出的第一个token的前10个概率: {dec_output[0, 0, :10]}")


if __name__ == "__main__":
    main()
