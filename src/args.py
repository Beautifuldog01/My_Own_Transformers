import argparse


def get_args():
    """
    解析命令行参数。
    返回：
    - 解析后的参数
    """
    parser = argparse.ArgumentParser(description="Transformer模型训练参数")

    # 模型参数
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="模型维度",
    )
    parser.add_argument(
        "--d_ff",
        type=int,
        default=2048,
        help="前馈网络维度",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="注意力头数",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=6,
        help="编码器和解码器层数",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout率",
    )

    # 训练参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批次大小",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="训练轮数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.0,
        help="学习率因子",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=4000,
        help="预热步数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="训练设备",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="模型保存目录",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="日志打印间隔",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="检查点文件路径，用于恢复训练",
    )
    parser.add_argument(
        "--save_best_only",
        action="store_true",
        help="是否只保存最佳模型",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="早停耐心值",
    )

    # 数据参数
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="训练数据文件路径",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="验证数据文件路径",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="最大序列长度",
    )
    parser.add_argument(
        "--use_demo_data",
        action="store_true",
        help="是否使用示例数据",
    )

    return parser.parse_args()
