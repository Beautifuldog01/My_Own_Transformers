import torch
import torch.nn as nn
import os
from tqdm import tqdm
from models import Encoder, Decoder
from config import config
from optimizer import get_optimizer
from utils import (
    get_tokenizer,
    encode_text,
    decode_text,
    create_masks,
    evaluate_translations,
    get_demo_data,
    load_data_from_file,
)


class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            device=config.device,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            device=config.device,
        )
        self.device = config.device

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output

    def translate(self, src, tokenizer, max_length=50):
        """
        翻译单个句子。
        参数：
        - src: 源序列
        - tokenizer: 分词器
        - max_length: 最大生成长度
        返回：
        - 翻译结果
        """
        self.eval()
        with torch.no_grad():
            # 创建源序列掩码
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

            # 编码源序列
            enc_output = self.encoder(src, src_mask)

            # 初始化目标序列
            tgt = (
                torch.ones(1, 1)
                .fill_(tokenizer.encode("<|startoftext|>")[0])
                .long()
                .to(self.device)
            )

            # 自回归生成
            for _ in range(max_length):
                # 创建目标序列掩码
                tgt_mask = create_masks(tgt, tgt)[1]

                # 解码
                output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

                # 获取下一个token
                next_token = output[:, -1].argmax(dim=-1).unsqueeze(0)

                # 如果生成了结束token，停止生成
                if next_token.item() == tokenizer.encode("<|endoftext|>")[0]:
                    break

                # 将新token添加到目标序列
                tgt = torch.cat([tgt, next_token], dim=1)

            # 解码生成的序列
            translation = decode_text(tgt[0], tokenizer)
            return translation


def train_step(model, optimizer, criterion, src, tgt, src_mask, tgt_mask):
    """
    执行一个训练步骤。
    参数：
    - model: Transformer模型
    - optimizer: 优化器
    - criterion: 损失函数
    - src: 源序列
    - tgt: 目标序列
    - src_mask: 源序列掩码
    - tgt_mask: 目标序列掩码
    返回：
    - 损失值
    """
    # 前向传播
    output = model(src, tgt, src_mask, tgt_mask)

    # 计算损失
    # 将输出和目标序列调整为适合计算损失的形式
    output = output.view(-1, output.size(-1))
    tgt = tgt.view(-1)
    loss = criterion(output, tgt)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, tokenizer, eval_data):
    """
    评估模型性能。
    参数：
    - model: Transformer模型
    - tokenizer: 分词器
    - eval_data: 评估数据
    返回：
    - 平均BLEU评分
    """
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for src, tgt in eval_data:
            # 将数据移到设备上
            src = src.to(config.device)

            # 生成翻译
            translation = model.translate(src, tokenizer)

            # 获取参考翻译
            reference = decode_text(tgt[0], tokenizer)

            references.append(reference)
            hypotheses.append(translation)

    # 计算BLEU评分
    bleu_score = evaluate_translations(references, hypotheses)
    return bleu_score


def train(model, optimizer, criterion, train_data, val_data, tokenizer):
    """
    训练模型。
    参数：
    - model: Transformer模型
    - optimizer: 优化器
    - criterion: 损失函数
    - train_data: 训练数据
    - val_data: 验证数据
    - tokenizer: 分词器
    """
    # 创建保存目录
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # 初始化最佳BLEU分数
    best_bleu = 0.0
    patience_counter = 0

    model.train()
    for epoch in tqdm(range(config.num_epochs), desc="Training"):
        total_loss = 0
        for i, (src, tgt) in enumerate(train_data):
            # 创建掩码
            src_mask, tgt_mask = create_masks(src, tgt)

            # 将数据移到设备上
            src = src.to(config.device)
            tgt = tgt.to(config.device)
            src_mask = src_mask.to(config.device)
            tgt_mask = tgt_mask.to(config.device)

            # 训练步骤
            loss = train_step(model, optimizer, criterion, src, tgt, src_mask, tgt_mask)
            total_loss += loss

            # 打印训练信息
            if (i + 1) % config.log_interval == 0:
                avg_loss = total_loss / config.log_interval
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs}, "
                    f"Batch {i + 1}/{len(train_data)}, "
                    f"Loss: {avg_loss:.4f}"
                )
                total_loss = 0

        # 评估模型
        train_bleu = evaluate(model, tokenizer, train_data)
        val_bleu = evaluate(model, tokenizer, val_data)
        print(
            f"Epoch {epoch + 1}/{config.num_epochs}, "
            f"Train BLEU: {train_bleu:.4f}, "
            f"Val BLEU: {val_bleu:.4f}"
        )

        # 保存模型
        if not config.save_best_only or val_bleu > best_bleu:
            # 更新最佳BLEU分数
            if val_bleu > best_bleu:
                best_bleu = val_bleu
                patience_counter = 0
            else:
                patience_counter += 1

            # 保存检查点
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_bleu": train_bleu,
                "val_bleu": val_bleu,
                "best_bleu": best_bleu,
            }
            torch.save(
                checkpoint,
                os.path.join(config.save_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
            )

        # 早停检查
        if patience_counter >= config.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载检查点。
    参数：
    - model: Transformer模型
    - optimizer: 优化器
    - checkpoint_path: 检查点文件路径
    返回：
    - 起始轮数
    - 最佳BLEU分数
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["best_bleu"]


def main():
    # 打印配置信息
    print(config)

    # 初始化分词器
    tokenizer = get_tokenizer()

    # 加载数据
    if config.train_file and os.path.exists(config.train_file):
        print(f"从文件加载训练数据: {config.train_file}")
        train_data = load_data_from_file(config.train_file, tokenizer)
        val_data = (
            load_data_from_file(config.val_file, tokenizer) if config.val_file else []
        )
    else:
        print("使用示例训练数据")
        train_data, val_data, test_data = get_demo_data(tokenizer)

    # 创建模型
    model = Transformer(vocab_size=tokenizer.n_vocab)

    # 创建优化器
    optimizer = get_optimizer(
        model=model,
        model_size=config.d_model,
        factor=config.learning_rate,
        warmup=config.warmup_steps,
    )

    # 创建损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充token

    # 如果提供了检查点，加载它
    start_epoch = 0
    best_bleu = 0.0
    if config.resume and os.path.exists(config.resume):
        print(f"从检查点恢复训练: {config.resume}")
        start_epoch, best_bleu = load_checkpoint(model, optimizer, config.resume)
        print(f"从第 {start_epoch} 轮继续训练，最佳BLEU分数: {best_bleu:.4f}")

    # 训练模型
    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
    )

    # 测试模型
    if not config.train_file or not config.val_file:
        print("\n测试模型性能:")
        test_bleu = evaluate(model, tokenizer, test_data)
        print(f"测试集BLEU分数: {test_bleu:.4f}")


if __name__ == "__main__":
    main()
