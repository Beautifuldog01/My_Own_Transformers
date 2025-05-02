import torch
from args import get_args


class Config:
    def __init__(self):
        # 解析命令行参数
        args = get_args()

        # 模型参数
        self.d_model = args.d_model  # 模型维度
        self.d_ff = args.d_ff  # 前馈网络维度
        self.num_heads = args.num_heads  # 注意力头数
        self.num_layers = args.num_layers  # 编码器和解码器层数
        self.dropout = args.dropout  # dropout率

        # 训练参数
        self.batch_size = args.batch_size  # 批次大小
        self.num_epochs = args.num_epochs  # 训练轮数
        self.learning_rate = args.learning_rate  # 学习率
        self.warmup_steps = args.warmup_steps  # 预热步数
        self.device = args.device if torch.cuda.is_available() else "cpu"  # 设备选择
        self.save_dir = args.save_dir  # 模型保存目录
        self.log_interval = args.log_interval  # 日志打印间隔
        self.resume = args.resume
        self.save_best_only = args.save_best_only
        self.patience = args.patience

        # 数据参数
        self.train_file = args.train_file  # 训练数据文件路径
        self.val_file = args.val_file  # 验证数据文件路径
        self.max_seq_len = args.max_seq_len  # 最大序列长度
        self.use_demo_data = args.use_demo_data

        # 设置设备
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA不可用，使用CPU")
            self.device = "cpu"

    def __str__(self):
        """返回配置的字符串表示"""
        config_str = "模型配置:\n"
        config_str += f"  d_model: {self.d_model}\n"
        config_str += f"  d_ff: {self.d_ff}\n"
        config_str += f"  num_heads: {self.num_heads}\n"
        config_str += f"  num_layers: {self.num_layers}\n"
        config_str += f"  dropout: {self.dropout}\n"
        config_str += "\n训练配置:\n"
        config_str += f"  batch_size: {self.batch_size}\n"
        config_str += f"  num_epochs: {self.num_epochs}\n"
        config_str += f"  learning_rate: {self.learning_rate}\n"
        config_str += f"  warmup_steps: {self.warmup_steps}\n"
        config_str += f"  device: {self.device}\n"
        config_str += f"  save_dir: {self.save_dir}\n"
        config_str += f"  log_interval: {self.log_interval}\n"
        config_str += f"  resume: {self.resume}\n"
        config_str += f"  save_best_only: {self.save_best_only}\n"
        config_str += f"  patience: {self.patience}\n"
        config_str += "\n数据配置:\n"
        config_str += f"  train_file: {self.train_file}\n"
        config_str += f"  val_file: {self.val_file}\n"
        config_str += f"  max_seq_len: {self.max_seq_len}\n"
        config_str += f"  use_demo_data: {self.use_demo_data}\n"
        return config_str


# 创建全局配置实例
config = Config()
