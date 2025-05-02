import torch


class Config:
    # 模型参数
    d_model = 512  # 模型维度
    d_ff = 2048  # 前馈网络维度
    num_heads = 8  # 注意力头数
    num_layers = 6  # 编码器和解码器层数
    dropout = 0.1  # dropout率

    # 训练参数
    device = "cuda" if torch.cuda.is_available() else "cpu"  #  设备选择
