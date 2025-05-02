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
)
from data import create_dataloader, load_data_from_file


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
            # 只翻译第一个句子（批次中的第一个）
            single_src = src[0:1]

            # 将源序列移到设备上
            single_src = single_src.to(self.device)

            # 创建源序列掩码
            src_mask = (single_src != 0).unsqueeze(1).unsqueeze(2).to(self.device)

            # 编码源序列
            enc_output = self.encoder(single_src, src_mask)

            # 初始化目标序列 - 使用一个数字而不是特殊标记
            tgt = torch.ones(1, 1).long().to(self.device)

            # 自回归生成
            for _ in range(max_length):
                # 创建目标序列掩码
                tgt_len = tgt.size(1)

                # 创建自回归掩码（防止看到未来tokens）
                look_ahead_mask = (
                    torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1)
                    .bool()
                    .to(self.device)
                )
                tgt_mask = ~look_ahead_mask  # 反转掩码（1表示注意，0表示忽略）
                tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(
                    0
                )  # [1, 1, tgt_len, tgt_len]

                # 解码
                output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

                # 获取下一个token
                next_token = output[:, -1].argmax(dim=-1).unsqueeze(1)

                # 如果生成了一个特定的结束标记，停止生成（使用一个罕见的标记作为结束标记）
                if next_token.item() == 100:  # 选择一个罕见的标记作为结束
                    break

                # 将新token添加到目标序列
                tgt = torch.cat([tgt, next_token], dim=1)

                # 防止生成过长
                if tgt.size(1) >= max_length:
                    break

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
        # 只评估前5个样本以加快速度
        for i, (src, tgt) in enumerate(eval_data):
            if i >= 5:  # 限制评估的样本数量
                break

            # 打印数据形状，帮助调试
            print(f"样本 {i+1}: Source shape: {src.shape}, Target shape: {tgt.shape}")

            try:
                # 将数据移到设备上
                src = src.to(model.device)
                tgt = tgt.to(model.device)

                # 生成翻译
                translation = model.translate(src, tokenizer)
                print(f"生成的翻译: {translation}")

                # 获取参考翻译
                reference = decode_text(tgt[0], tokenizer)
                print(f"参考翻译: {reference}")

                # 记录翻译结果
                references.append(reference)
                hypotheses.append(translation)
                print(f"成功翻译样本 {i+1}")

            except Exception as e:
                print(f"翻译样本 {i+1} 时出错: {e}")
                import traceback

                traceback.print_exc()
                continue

    # 如果没有成功的翻译，返回0
    if len(references) == 0:
        print("警告: 没有成功的翻译")
        return 0.0

    # 计算BLEU评分
    print(f"参考翻译: {references}")
    print(f"模型翻译: {hypotheses}")
    bleu_score = evaluate_translations(references, hypotheses)
    return bleu_score


def train(model, optimizer, criterion, train_loader, val_loader, tokenizer):
    """
    训练模型。
    参数：
    - model: Transformer模型
    - optimizer: 优化器
    - criterion: 损失函数
    - train_loader: 训练数据加载器
    - val_loader: 验证数据加载器
    - tokenizer: 分词器
    """
    # 创建保存目录
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # 初始化最佳BLEU分数
    best_bleu = 0.0
    patience_counter = 0

    # 尝试执行一个评估，确保一切正常
    print("初始评估...")
    try:
        # 仅评估一个小批次
        initial_src, initial_tgt = next(iter(train_loader))
        initial_src = initial_src[:1].to(model.device)
        initial_tgt = initial_tgt[:1].to(model.device)

        # 创建掩码
        src_mask = (initial_src != 0).unsqueeze(1).unsqueeze(2).to(model.device)
        tgt_len = initial_tgt.size(1)
        tgt_mask = (
            torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(model.device)
        )
        tgt_mask = ~tgt_mask
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

        # 尝试前向传播
        with torch.no_grad():
            output = model(initial_src, initial_tgt, src_mask, tgt_mask)
        print("初始评估成功!")
    except Exception as e:
        print(f"初始评估失败: {e}")
        raise e

    model.train()
    for epoch in tqdm(range(config.num_epochs), desc="Training"):
        total_loss = 0
        for i, (src, tgt) in enumerate(train_loader):
            # 将数据移到设备上
            src = src.to(model.device)
            tgt = tgt.to(model.device)

            # 创建掩码
            # 源序列掩码 [batch_size, 1, 1, src_len]
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(model.device)

            # 目标序列掩码 [batch_size, 1, tgt_len, tgt_len]
            tgt_len = tgt.size(1)

            # 创建前瞻掩码
            tgt_mask = (
                torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1)
                .bool()
                .to(model.device)
            )
            tgt_mask = ~tgt_mask  # 反转掩码
            tgt_mask = (
                tgt_mask.unsqueeze(0)
                .unsqueeze(0)
                .expand(src.size(0), 1, tgt_len, tgt_len)
            )

            # 训练步骤
            loss = train_step(model, optimizer, criterion, src, tgt, src_mask, tgt_mask)
            total_loss += loss

            # 打印训练信息
            if (i + 1) % config.log_interval == 0:
                avg_loss = total_loss / config.log_interval
                print(
                    f"Epoch {epoch + 1}/{config.num_epochs}, "
                    f"Batch {i + 1}/{len(train_loader)}, "
                    f"Loss: {avg_loss:.4f}"
                )
                total_loss = 0

        # 评估模型
        print("评估训练集BLEU分数...")
        train_bleu = evaluate(model, tokenizer, train_loader)
        print("评估验证集BLEU分数...")
        val_bleu = evaluate(model, tokenizer, val_loader)
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

    # 获取分词器
    tokenizer = get_tokenizer()

    # 准备数据
    if config.use_demo_data or not config.train_file:
        print("使用示例数据")
        train_data, val_data, test_data = get_demo_data(tokenizer)
        train_loader = create_dataloader(
            train_data, config.batch_size, config.max_seq_len
        )
        val_loader = create_dataloader(val_data, config.batch_size, config.max_seq_len)
        test_loader = create_dataloader(
            test_data, config.batch_size, config.max_seq_len
        )
    else:
        print(f"从文件加载训练数据: {config.train_file}")
        train_loader = load_data_from_file(
            config.train_file, config.batch_size, config.max_seq_len
        )
        val_loader = (
            load_data_from_file(config.val_file, config.batch_size, config.max_seq_len)
            if config.val_file
            else None
        )
        test_loader = None

    # 创建模型
    model = Transformer(vocab_size=tokenizer.n_vocab).to(config.device)

    # 创建优化器和损失函数
    optimizer = get_optimizer(
        model=model,
        model_size=config.d_model,
        factor=config.learning_rate,
        warmup=config.warmup_steps,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 如果指定了检查点，加载它
    if config.resume:
        load_checkpoint(model, optimizer, config.resume)

    # 训练模型
    train(model, optimizer, criterion, train_loader, val_loader, tokenizer)

    # 如果使用了示例数据，在测试集上评估模型
    if test_loader is not None:
        test_bleu = evaluate(model, tokenizer, test_loader)
        print(f"Test BLEU: {test_bleu:.4f}")


if __name__ == "__main__":
    main()
