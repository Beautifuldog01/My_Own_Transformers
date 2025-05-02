# My_Own_Transformers

我自己的Transformer模型实现。

本仓库包含对原始Transformer论文（"Attention Is All You Need"）的实现。作者将持续更新代码以匹配论文的细节。

<div align="center">
  <img src="https://github.com/user-attachments/assets/1d549173-450a-484a-af29-47152805800d" width="50%">
</div>

## 项目结构

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

## 安装

1.  **克隆仓库:**
    ```bash
    git clone https://github.com/your-username/My_Own_Transformers.git
    cd My_Own_Transformers
    ```
2.  **安装依赖:**
    ```bash
    pip install -r requirements.txt
    ```

## 使用方法

### 数据准备

训练数据应该是制表符分隔的源语言和目标语言句子对，每行一对。例如：

```
Learning is the best reward.    学习是旅途的意义。
Knowledge is power.    知识就是力量。
Practice makes perfect.    熟能生巧。
```

### 数据加载

该项目支持两种数据加载方式：

1. **使用内置示例数据:** 代码提供了一组英汉翻译对作为示例数据。
2. **自定义数据文件:** 可以指定自己的翻译数据文件路径。

数据加载模块通过`data.py`中的`TranslationDataset`类和`create_dataloader`函数处理数据，支持批处理和填充等操作。

### 训练模型

你有两种选择来训练模型：

1. **使用内置示例数据:**

   直接运行脚本，无需指定训练文件 - 代码将使用内置的示例翻译：

   ```bash
   python src/main.py --use_demo_data
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

## 实现细节

### 多头注意力机制

多头注意力机制在`models.py`中的`MultiHeadAttention`类实现，包括：

1. **自注意力**: 编码器中使用，对输入序列进行自注意力计算。
2. **掩码自注意力**: 解码器中使用，确保解码时只能看到当前及之前的位置。
3. **编码器-解码器注意力**: 解码器中使用，将解码器的查询与编码器的键和值进行注意力计算。

掩码处理经过优化，确保在不同的注意力类型中维度正确匹配。

### 优化器与学习率调度

Transformer使用自定义的学习率调度策略，在`optimizer.py`中实现。学习率计算公式为：

```
lr = factor * (d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5)))
```

这种调度策略在训练初期学习率逐渐增大，之后缓慢减小，帮助模型更好地收敛。

### BLEU评分计算

使用`utils.py`中的`calculate_bleu`和`evaluate_translations`函数实现BLEU分数计算，以评估翻译质量。为了更好地支持中文，BLEU计算是在字符级别进行的。

## 主要特性

*   **多头注意力**: 完整实现论文中的多头注意力机制，支持自注意力和编码器-解码器注意力。
*   **位置编码**: 使用正弦和余弦函数实现位置编码，提供序列位置信息。
*   **残差连接**: 每个子层之后都有残差连接和层归一化，帮助训练深层网络。
*   **自定义优化器**: 实现原始论文中的学习率调度策略。
*   **掩码机制**: 实现填充掩码和前瞻掩码，处理变长序列和保证自回归特性。
*   **BLEU评估**: 使用字符级BLEU分数评估翻译质量。
*   **检查点管理**: 保存和恢复训练状态，支持断点续训。
*   **早停机制**: 通过监控验证集BLEU分数防止过拟合。
*   **数据加载器**: 使用PyTorch的Dataset和DataLoader进行高效的数据加载。
*   **批处理与填充**: 自动处理变长序列的批处理。
*   **设备适配**: 自动检测可用设备，支持CPU和CUDA训练。

## 最近更新

*   **修复掩码维度问题**: 解决了注意力计算中掩码维度不匹配的问题。
*   **优化器状态管理**: 完善了优化器的状态保存和加载。
*   **改进评估逻辑**: 增强了翻译评估过程，添加更详细的调试信息。
*   **增强数据处理**: 改进了数据加载和预处理流程，更好地处理批处理和填充。
*   **健壮的错误处理**: 增加了更完善的异常处理，提高代码健壮性。

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
