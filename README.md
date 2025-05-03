# My_Own_Transformers


This repository contains an implementation of the original Transformer paper ("Attention Is All You Need"). We will continuously update the code to match the details of the paper.

Also, about this repository, I have a tiny blog about interesting details when reproducing the implementation, you can visit the site by click[2025年还有人在Attention Is All You Need - 嘻嘻福斯的文章 - 知乎](https://zhuanlan.zhihu.com/p/1901912964216358105)

<div align="center">
  <img src="https://github.com/user-attachments/assets/1d549173-450a-484a-af29-47152805800d" width="50%">
</div>

## Project Structure

```
src/
├── models.py       # Transformer model implementation (Encoder, Decoder, etc.)
├── optimizer.py    # Custom optimizer with learning rate scheduling
├── utils.py        # Utility functions (masking, tokenization, evaluation, etc.)
├── config.py       # Configuration management (loaded from args)
├── args.py         # Command line argument parsing
├── main.py         # Main script for training and evaluation
└── data.py         # Data loading and preprocessing
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/My_Own_Transformers.git
    cd My_Own_Transformers
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preparation

Training data should be tab-separated pairs of source and target language sentences, one pair per line. For example:

```
Learning is the best reward.    学习是旅途的意义。
Knowledge is power.    知识就是力量。
Practice makes perfect.    熟能生巧。
```

### Data Loading

The project supports two ways to load data:

1. **Using built-in demo data:** The code provides a set of English-Chinese translation pairs as demo data.
2. **Custom data files:** You can specify your own translation data file path.

The data loading module handles data through the `TranslationDataset` class and `create_dataloader` function in `data.py`, supporting batching and padding operations.

### Model Training

You have two options to train the model:

1. **Using built-in demo data:**

   Run the script directly without specifying a training file - the code will use the built-in demo translations:

   ```bash
   python src/main.py --use_demo_data
   ```

2. **Using custom training data:**

   Specify your training data file with the `--train_file` parameter:

   ```bash
   python src/main.py --train_file path/to/your/train_data.txt
   ```

### Model Evaluation

The model is evaluated using BLEU scores on the training and validation sets after each epoch. You can customize the evaluation process with the following options:

*   `--val_file` (str, default: None): Path to the validation data file.
*   `--save_best_only` (flag): Save only the best model based on validation BLEU score.
*   `--patience` (int, default: 5): Number of epochs to wait for improvement in validation BLEU score before stopping training.

### Resuming Training

You can resume training from a checkpoint using the `--resume` parameter:

```bash
python src/main.py --resume path/to/checkpoint.pt
```

### Full Example

Here is a full example with all parameters:

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

### Command Line Arguments

**Model Parameters:**

*   `--d_model` (int, default: 512): Model dimension.
*   `--d_ff` (int, default: 2048): Feedforward network dimension.
*   `--num_heads` (int, default: 8): Number of attention heads.
*   `--num_layers` (int, default: 6): Number of encoder and decoder layers.
*   `--dropout` (float, default: 0.1): Dropout rate.

**Training Parameters:**

*   `--batch_size` (int, default: 32): Batch size.
*   `--num_epochs` (int, default: 10): Number of training epochs.
*   `--learning_rate` (float, default: 1.0): Learning rate factor for the optimizer.
*   `--warmup_steps` (int, default: 4000): Number of warmup steps for learning rate.
*   `--device` (str, default: "cuda"): Training device ("cuda" or "cpu"). Defaults to "cpu" if CUDA is unavailable.
*   `--save_dir` (str, default: "checkpoints"): Directory to save models.
*   `--log_interval` (int, default: 100): Interval for logging (in batches).
*   `--resume` (str, default: None): Path to checkpoint file for resuming training.
*   `--save_best_only` (flag): Whether to save only the best model.
*   `--patience` (int, default: 5): Patience for early stopping.

**Data Parameters:**

*   `--train_file` (str, default: None): Path to the training data file.
*   `--val_file` (str, default: None): Path to the validation data file.
*   `--max_seq_len` (int, default: 512): Maximum sequence length.
*   `--use_demo_data` (flag): Whether to use demo data.

## Implementation Details

### Multi-Head Attention Mechanism

The multi-head attention mechanism is implemented in the `MultiHeadAttention` class in `models.py`, including:

1. **Self-Attention**: Used in the encoder for self-attention computation on the input sequence.
2. **Masked Self-Attention**: Used in the decoder to ensure that decoding can only see the current and previous positions.
3. **Encoder-Decoder Attention**: Used in the decoder to compute attention between the decoder's queries and the encoder's keys and values.

Masking is optimized to ensure correct dimension matching in different attention types.

### Optimizer and Learning Rate Scheduling

The Transformer uses a custom learning rate scheduling strategy implemented in `optimizer.py`. The learning rate is calculated as:

```
lr = factor * (d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5)))
```

This scheduling strategy increases the learning rate gradually at the beginning of training and then decreases it slowly, helping the model converge better.

### BLEU Score Calculation

BLEU score calculation is implemented using the `calculate_bleu` and `evaluate_translations` functions in `utils.py` to evaluate translation quality. To better support Chinese, BLEU calculation is performed at the character level.

## Key Features

*   **Multi-Head Attention**: Complete implementation of the multi-head attention mechanism from the paper, supporting self-attention and encoder-decoder attention.
*   **Positional Encoding**: Implemented using sine and cosine functions to provide positional information to the sequence.
*   **Residual Connections**: Each sub-layer is followed by a residual connection and layer normalization to help train deep networks.
*   **Custom Optimizer**: Implements the learning rate scheduling strategy from the original paper.
*   **Masking Mechanism**: Implements padding and look-ahead masks to handle variable-length sequences and ensure autoregressive properties.
*   **BLEU Evaluation**: Uses character-level BLEU scores to evaluate translation quality.
*   **Checkpoint Management**: Saves and restores training state, supporting resumption of training.
*   **Early Stopping**: Prevents overfitting by monitoring validation BLEU scores.
*   **Data Loader**: Efficient data loading using PyTorch's Dataset and DataLoader.
*   **Batching and Padding**: Automatically handles batching of variable-length sequences.
*   **Device Adaptation**: Automatically detects available devices, supporting both CPU and CUDA training.

## Recent Updates

*   **Fixed Mask Dimension Issues**: Resolved issues with mask dimension mismatches in attention calculations.
*   **Optimizer State Management**: Improved saving and loading of optimizer state.
*   **Improved Evaluation Logic**: Enhanced the translation evaluation process with more detailed debugging information.
*   **Enhanced Data Processing**: Improved data loading and preprocessing to better handle batching and padding.
*   **Robust Error Handling**: Added more comprehensive exception handling to improve code robustness.

## Checkpoint Files

Checkpoint files contain the following information:
*   `epoch`: Current training epoch
*   `model_state_dict`: Model state dictionary
*   `optimizer_state_dict`: Optimizer state dictionary
*   `train_bleu`: Training set BLEU score
*   `val_bleu`: Validation set BLEU score
*   `best_bleu`: Best validation set BLEU score

## Special Token Handling

In this implementation, special tokens are handled using reserved IDs:
*   `PAD_ID = 0`: Used for padding sequences to the same length.
*   `BOS_ID = 1`: Used to indicate the beginning of a sequence.
*   `EOS_ID = 2`: Used to indicate the end of a sequence.

These IDs are used consistently across tokenization, decoding, masking, and padding processes to ensure the model correctly learns when to start, end, and pad sequences.

## Citation

If you find this implementation helpful, please consider citing the original paper:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
