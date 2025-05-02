import torch
import torch.nn as nn
import os
from models import Encoder, Decoder
from config import config
from optimizer import get_optimizer
from utils import get_tokenizer, encode_text, create_masks


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


def train(model, optimizer, criterion, train_data):
    """
    训练模型。
    参数：
    - model: Transformer模型
    - optimizer: 优化器
    - criterion: 损失函数
    - train_data: 训练数据
    """
    model.train()
    for epoch in range(config.num_epochs):
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

        # 保存模型
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        torch.save(
            model.state_dict(),
            os.path.join(config.save_dir, f"model_epoch_{epoch + 1}.pt"),
        )


def main():
    # 打印配置信息
    print(config)

    # 初始化分词器
    tokenizer = get_tokenizer()

    # 示例训练数据
    train_data = [
        ("Learning is the best reward.", "学习是旅途的意义。"),
        ("Knowledge is power.", "知识就是力量。"),
        ("Practice makes perfect.", "熟能生巧。"),
    ]

    # 将文本转换为token
    train_data = [
        (encode_text(src, tokenizer), encode_text(tgt, tokenizer))
        for src, tgt in train_data
    ]

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

    # 训练模型
    train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_data=train_data,
    )


if __name__ == "__main__":
    main()
