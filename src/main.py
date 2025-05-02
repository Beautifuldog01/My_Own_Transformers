import torch
import tiktoken
from models import Encoder
from config import Config


def main():
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # 示例文本
    text = "这是一个测试句子，用于演示Transformer的encoder部分。"

    # 使用分词器编码文本
    tokens = tokenizer.encode(text)
    tokens = torch.tensor(tokens).unsqueeze(0)  # 添加batch维度

    # 创建模型
    encoder = Encoder(
        vocab_size=tokenizer.n_vocab,
        d_model=Config.d_model,
        num_heads=Config.num_heads,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        dropout=Config.dropout,
    )

    # 前向传播
    output = encoder(tokens)

    print(f"输入文本: {text}")
    print(f"输入tokens: {tokens}")
    print(f"输出形状: {output.shape}")
    print(f"输出示例: {output[0, 0, :5]}")  # 打印第一个token的前5个维度


if __name__ == "__main__":
    main()
