class Config:
    # 模型参数
    d_model = 512  # 模型维度
    d_ff = 2048  # 前馈网络维度
    num_heads = 8  # 注意力头数
    num_layers = 6  # 编码器和解码器层数
    dropout = 0.1  # dropout率

    # 训练参数
    batch_size = 64
    learning_rate = 0.0001
    warmup_steps = 4000
    max_seq_length = 512

    # 词汇表大小
    vocab_size = 37000  # 可以根据实际数据集调整

    # 其他参数
    label_smoothing = 0.1
    beam_size = 4
    max_length = 100
