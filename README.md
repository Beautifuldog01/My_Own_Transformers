# My_Own_Transformers

My own and noob implementation of Transformer Model Family.

This repository contains an ongoing implementation of the original Transformer paper ("Attention Is All You Need"). The author will continuously update the code to match the paper's details.

<div align="center">
  <img src="https://github.com/user-attachments/assets/1d549173-450a-484a-af29-47152805800d" width="50%">
</div>

## Project Structure

```
src/
├── models.py       # Transformer模型实现 (Encoder, Decoder等)
├── optimizer.py    # 自定义优化器，包含学习率调度
├── utils.py        # 工具函数 (掩码、分词、评估等)
├── config.py       # 配置管理 (从args加载)
├── args.py         # 命令行参数解析
├── main.py         # 训练和评估的主脚本
└── data.py         # 数据加载和预处理
```

## Installation

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/your-username/My_Own_Transformers.git
    cd My_Own_Transformers
    ```
2.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 数据准备

训练数据应该是制表符分隔的源语言和目标语言句子对，每行一对。例如：

```
Learning is the best reward.    学习是旅途的意义。
Knowledge is power.    知识就是力量。
Practice makes perfect.    熟能生巧。
```

### 训练模型

你有两种选择来训练模型：

1. **使用内置示例数据:**

   直接运行脚本，无需指定训练文件 - 代码将使用内置的示例翻译：

   ```bash
   python src/main.py
   ```

2. **使用自定义训练数据:**

   使用`--train_file`参数指定你的训练数据文件：

   ```bash
   python src/main.py --train_file path/to/your/train_data.txt
   ```

### 模型评估

模型在每个epoch后使用BLEU分数在训练集和验证集上进行评估。你可以通过以下选项自定义评估过程：

*   `--val_file` (str, default: None): 验证数据文件路径。
*   `--save_best_only` (flag): 只保存基于验证集BLEU分数的最佳模型。
*   `--patience` (int, default: 5): 如果验证集BLEU分数没有改善，等待多少个epoch后停止训练。

### 恢复训练

你可以使用`--resume`参数从检查点恢复训练：

```bash
python src/main.py --resume path/to/checkpoint.pt
```

### 完整示例

以下是一个包含所有参数的完整示例：

```bash
python src/main.py \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 1.0 \
    --warmup_steps 4000 \
    --save_dir checkpoints \
    --train_file data/train.txt \
    --val_file data/val.txt \
    --save_best_only \
    --patience 5 \
    --resume checkpoints/checkpoint_epoch_5.pt
```

### 命令行参数

**模型参数:**

*   `--d_model` (int, default: 512): 模型维度。
*   `--d_ff` (int, default: 2048): 前馈网络维度。
*   `--num_heads` (int, default: 8): 注意力头数。
*   `--num_layers` (int, default: 6): 编码器和解码器层数。
*   `--dropout` (float, default: 0.1): Dropout率。

**训练参数:**

*   `--batch_size` (int, default: 32): 批次大小。
*   `--num_epochs` (int, default: 10): 训练轮数。
*   `--learning_rate` (float, default: 1.0): 优化器学习率因子。
*   `--warmup_steps` (int, default: 4000): 学习率预热步数。
*   `--device` (str, default: "cuda"): 训练设备 ("cuda" 或 "cpu")。如果CUDA不可用，将默认使用"cpu"。
*   `--save_dir` (str, default: "checkpoints"): 模型保存目录。
*   `--log_interval` (int, default: 100): 日志打印间隔（以批次为单位）。
*   `--resume` (str, default: None): 用于恢复训练的检查点文件路径。
*   `--save_best_only` (flag): 是否只保存最佳模型。
*   `--patience` (int, default: 5): 早停耐心值。

**数据参数:**

*   `--train_file` (str, default: None): 训练数据文件路径。
*   `--val_file` (str, default: None): 验证数据文件路径。
*   `--max_seq_len` (int, default: 512): 最大序列长度。
*   `--use_demo_data` (flag): 是否使用示例数据。

## 特性

*   **自定义优化器**: 实现了原始论文中的学习率调度。
*   **BLEU评估**: 使用字符级BLEU分数评估翻译质量。
*   **检查点管理**: 保存和恢复训练检查点。
*   **早停机制**: 通过监控验证集BLEU分数防止过拟合。
*   **掩码实现**: 实现了Transformer的填充掩码和前瞻掩码。
*   **示例数据**: 包含内置的示例翻译用于快速测试。
*   **数据加载器**: 使用PyTorch的Dataset和DataLoader进行高效的数据加载。
*   **批处理**: 支持批处理和随机打乱。
*   **序列截断**: 自动处理过长的序列。

## 检查点文件

检查点文件包含以下信息：
*   `epoch`: 当前训练轮数
*   `model_state_dict`: 模型状态字典
*   `optimizer_state_dict`: 优化器状态字典
*   `train_bleu`: 训练集BLEU分数
*   `val_bleu`: 验证集BLEU分数
*   `best_bleu`: 最佳验证集BLEU分数

## 引用

如果你觉得这个实现有帮助，请考虑引用原始论文：

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
