# My_Own_Transformers

My own and noob implementation of Transformer Model Family.

This repository contains an ongoing implementation of the original Transformer paper ("Attention Is All You Need"). The author will continuously update the code to match the paper's details.

<div align="center">
  <img src="https://github.com/user-attachments/assets/1d549173-450a-484a-af29-47152805800d" width="50%">
</div>

## Project Structure

```
src/
├── models.py       # Transformer model implementation (Encoder, Decoder, etc.)
├── optimizer.py    # Custom optimizer with learning rate scheduling
├── utils.py        # Utility functions (masking, tokenization, evaluation)
├── config.py       # Configuration management (loads from args)
├── args.py         # Command-line argument parsing
├── main.py         # Main script for training and evaluation
└── data.py         # (Placeholder for data loading/preprocessing)
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

### Training the Model

You have two options for training data:

1. **Using Built-in Demo Data:**

   Simply run the script without specifying a training file - the code will use the built-in example translations:

   ```bash
   python src/main.py
   ```

2. **Using Custom Training Data:**

   Specify your training data file with the `--train_file` parameter. The file should contain tab-separated source and target sentences, one pair per line:

   ```bash
   python src/main.py --train_file path/to/your/train_data.txt
   ```

### Model Evaluation

The model is evaluated using BLEU score on both training and validation sets after each epoch. You can customize the evaluation process with the following options:

*   `--val_file` (str, default: None): Path to the validation data file.
*   `--save_best_only` (flag): Save only the best model based on validation BLEU score.
*   `--patience` (int, default: 5): Number of epochs to wait before early stopping if validation BLEU doesn't improve.

### Resuming Training

You can resume training from a checkpoint using the `--resume` parameter:

```bash
python src/main.py --resume path/to/checkpoint.pt
```

### Complete Example

Here's a complete example with all parameters:

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

### Command-Line Arguments

**Model Parameters:**

*   `--d_model` (int, default: 512): Model dimension.
*   `--d_ff` (int, default: 2048): Feed-forward network dimension.
*   `--num_heads` (int, default: 8): Number of attention heads.
*   `--num_layers` (int, default: 6): Number of encoder and decoder layers.
*   `--dropout` (float, default: 0.1): Dropout rate.

**Training Parameters:**

*   `--batch_size` (int, default: 32): Batch size.
*   `--num_epochs` (int, default: 10): Number of training epochs.
*   `--learning_rate` (float, default: 1.0): Learning rate factor for the optimizer schedule.
*   `--warmup_steps` (int, default: 4000): Number of warmup steps for the learning rate scheduler.
*   `--device` (str, default: "cuda"): Training device ("cuda" or "cpu"). Will default to "cpu" if cuda is not available.
*   `--save_dir` (str, default: "checkpoints"): Directory to save model checkpoints.
*   `--log_interval` (int, default: 100): Logging interval (in batches).
*   `--resume` (str, default: None): Path to checkpoint file to resume training.
*   `--save_best_only` (flag): Save only the best model based on validation BLEU score.
*   `--patience` (int, default: 5): Number of epochs to wait before early stopping.

**Data Parameters:**

*   `--train_file` (str, default: None): Path to the training data file.
*   `--val_file` (str, default: None): Path to the validation data file.
*   `--max_seq_len` (int, default: 512): Maximum sequence length.
*   `--use_demo_data` (flag): Explicitly use demo data for training, even if a training file is provided.

## Features

*   **Custom Optimizer**: Implements the learning rate schedule from the original paper.
*   **BLEU Evaluation**: Uses character-level BLEU score for translation quality assessment.
*   **Checkpoint Management**: Save and resume training from checkpoints.
*   **Early Stopping**: Prevents overfitting by monitoring validation BLEU score.
*   **Masking**: Implements padding and look-ahead masks for the Transformer.
*   **Demo Data**: Includes built-in example translations for quick testing.

## Citation

If you find this implementation helpful, please consider citing the original paper:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
