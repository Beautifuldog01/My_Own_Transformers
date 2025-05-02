import torch
import tiktoken
from typing import Tuple, List
import collections
import math
import os


def create_padding_mask(seq):
    """
    创建填充掩码。
    参数：
    - seq: 输入序列
    返回：
    - 填充掩码
    """
    # 获取序列的设备
    device = seq.device

    # 创建掩码
    mask = (seq == 0).unsqueeze(1).unsqueeze(2)
    return mask.to(device)


def create_look_ahead_mask(size):
    """
    创建前瞻掩码。
    参数：
    - size: 序列长度
    返回：
    - 前瞻掩码
    """
    # 创建掩码
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.to(torch.bool)


def create_masks(src, tgt):
    """
    创建源序列和目标序列的掩码。
    参数：
    - src: 源序列，形状为 [batch_size, seq_len]
    - tgt: 目标序列，形状为 [batch_size, seq_len]
    返回：
    - src_mask: 源序列掩码，形状为 [batch_size, 1, 1, seq_len]
    - tgt_mask: 目标序列掩码，形状为 [batch_size, 1, seq_len, seq_len]
    """
    # 获取序列的设备
    device = src.device

    # 创建源序列掩码
    src_mask = (src == 0).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

    # 创建目标序列掩码
    tgt_padding_mask = (
        (tgt == 0).unsqueeze(1).unsqueeze(2)
    )  # [batch_size, 1, 1, seq_len]
    tgt_look_ahead_mask = (
        torch.triu(torch.ones(tgt.size(1), tgt.size(1)), diagonal=1).bool().to(device)
    )  # [seq_len, seq_len]
    tgt_mask = tgt_padding_mask | tgt_look_ahead_mask.unsqueeze(
        0
    )  # [batch_size, 1, seq_len, seq_len]

    return src_mask, tgt_mask


def get_tokenizer() -> tiktoken.Encoding:
    """
    获取分词器。
    返回：
    - 分词器实例
    """
    return tiktoken.get_encoding("cl100k_base")


def encode_text(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """
    将文本编码为token。
    参数：
    - text: 输入文本
    - tokenizer: 分词器
    返回：
    - token张量，形状为 [1, seq_len]
    """
    try:
        # 使用基本方法编码
        tokens = tokenizer.encode(text)
    except Exception as e:
        print(f"编码错误: {e}")
        # 降级到空编码
        tokens = [1]  # 使用一个通用token
    return torch.tensor(tokens).unsqueeze(0)


def decode_text(tokens: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """
    将token解码为文本。
    参数：
    - tokens: token张量，形状为 [seq_len]
    - tokenizer: 分词器
    返回：
    - 解码后的文本
    """
    # 移除0和特殊token
    tokens = tokens.cpu().numpy().tolist()

    # 移除0（填充标记）
    filtered_tokens = [t for t in tokens if t != 0]

    try:
        # 使用基本方法解码
        return tokenizer.decode(filtered_tokens)
    except Exception as e:
        print(f"解码错误: {e}")
        return "翻译结果不可用"


def calculate_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """
    计算BLEU评分。
    参数：
    - reference: 参考翻译
    - hypothesis: 模型生成的翻译
    - max_n: 最大n-gram长度
    返回：
    - BLEU评分，范围[0, 1]
    """
    # 将文本分词为字符列表(对中文友好)
    ref_tokens = list(reference)
    hyp_tokens = list(hypothesis)

    # 如果生成的翻译为空，返回0
    if len(hyp_tokens) == 0:
        return 0.0

    # 计算n-gram精度
    precisions = []
    for n in range(1, min(max_n, len(hyp_tokens)) + 1):
        # 计算参考翻译中的n-gram
        ref_ngrams = collections.Counter()
        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i : i + n])
            ref_ngrams[ngram] += 1

        # 计算生成翻译中的n-gram
        hyp_ngrams = collections.Counter()
        for i in range(len(hyp_tokens) - n + 1):
            ngram = tuple(hyp_tokens[i : i + n])
            hyp_ngrams[ngram] += 1

        # 计算匹配的n-gram数量
        matches = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))

        # 计算精度
        precision = matches / max(1, len(hyp_tokens) - n + 1)
        precisions.append(precision)

    # 计算长度惩罚因子
    if len(hyp_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    else:
        brevity_penalty = 1.0

    # 计算BLEU评分
    if any(p > 0 for p in precisions):
        s = math.log(brevity_penalty)
        s += sum(math.log(p) if p > 0 else float("-inf") for p in precisions) / len(
            precisions
        )
        bleu = math.exp(s)
    else:
        bleu = 0.0

    return bleu


def evaluate_translations(references: List[str], hypotheses: List[str]) -> float:
    """
    评估一组翻译的BLEU评分。
    参数：
    - references: 参考翻译列表
    - hypotheses: 模型生成的翻译列表
    返回：
    - 平均BLEU评分
    """
    if len(references) != len(hypotheses):
        raise ValueError("参考翻译和生成翻译的数量必须相同")

    total_bleu = 0.0
    for ref, hyp in zip(references, hypotheses):
        total_bleu += calculate_bleu(ref, hyp)

    return total_bleu / len(references)


def get_demo_data(
    tokenizer: tiktoken.Encoding,
) -> Tuple[
    List[Tuple[str, str]],
    List[Tuple[str, str]],
    List[Tuple[str, str]],
]:
    """
    获取示例训练、验证和测试数据。
    参数：
    - tokenizer: 分词器
    返回：
    - 训练数据、验证数据和测试数据的元组，每个元素是(源文本, 目标文本)的元组
    """
    # 训练数据
    train_data = [
        ("Learning is the best reward.", "学习是旅途的意义。"),
        ("Knowledge is power.", "知识就是力量。"),
        ("Practice makes perfect.", "熟能生巧。"),
        ("Time is money.", "时间就是金钱。"),
        ("Where there is a will, there is a way.", "有志者事竟成。"),
        ("Actions speak louder than words.", "行动胜于言语。"),
        ("The early bird catches the worm.", "早起的鸟儿有虫吃。"),
        (
            "A journey of a thousand miles begins with a single step.",
            "千里之行，始于足下。",
        ),
        ("Failure is the mother of success.", "失败是成功之母。"),
        ("Rome was not built in a day.", "罗马不是一天建成的。"),
    ]

    # 验证数据
    val_data = [
        ("All roads lead to Rome.", "条条大路通罗马。"),
        ("Better late than never.", "亡羊补牢，为时未晚。"),
        ("Every cloud has a silver lining.", "黑暗中总有一线光明。"),
        ("A friend in need is a friend indeed.", "患难见真情。"),
        ("Honesty is the best policy.", "诚实是最好的策略。"),
    ]

    # 测试数据
    test_data = [
        ("The grass is always greener on the other side.", "这山望着那山高。"),
        ("Don't put all your eggs in one basket.", "不要把所有鸡蛋放在一个篮子里。"),
        ("When in Rome, do as the Romans do.", "入乡随俗。"),
        ("A penny saved is a penny earned.", "省一分就是赚一分。"),
        ("Birds of a feather flock together.", "物以类聚，人以群分。"),
    ]

    return train_data, val_data, test_data


def load_data_from_file(
    file_path: str, tokenizer: tiktoken.Encoding
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    从文件加载训练数据。
    参数：
    - file_path: 文件路径
    - tokenizer: 分词器
    返回：
    - 训练数据列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    train_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                src, tgt = line.split("\t")
                train_data.append(
                    (encode_text(src, tokenizer), encode_text(tgt, tokenizer))
                )
            except ValueError:
                print(f"跳过无效行: {line}")

    return train_data


def save_data_to_file(data: List[Tuple[str, str]], file_path: str) -> None:
    """
    将数据保存到文件。
    参数：
    - data: 数据列表，每个元素是(源文本, 目标文本)的元组
    - file_path: 文件路径
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for src, tgt in data:
            f.write(f"{src}\t{tgt}\n")
