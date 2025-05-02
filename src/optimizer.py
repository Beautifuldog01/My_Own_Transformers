import torch
import math
from torch.optim import Optimizer
from typing import Optional


class TransformerOptimizer(Optimizer):
    """
    Transformer优化器，实现了论文中的学习率调度策略。
    学习率计算公式：lrate = d_model^{-0.5} * min(step_num^{-0.5}, step_num * warmup_steps^{-1.5})
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        """
        初始化优化器。
        参数：
        - model_size: 模型维度
        - factor: 缩放因子
        - warmup: 预热步数
        - optimizer: 基础优化器
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self, closure: Optional[callable] = None) -> float:
        """
        更新参数和学习率。
        参数：
        - closure: 用于重新评估模型并返回损失的闭包
        返回：
        - 损失值
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        return self.optimizer.step(closure)

    def rate(self, step=None):
        """
        计算当前步数的学习率。
        参数：
        - step: 当前步数，如果为None则使用内部计数器
        返回：
        - 学习率
        """
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self, set_to_none=False):
        """
        清空梯度。
        参数：
        - set_to_none: 是否将梯度设置为None而不是零
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)


def get_optimizer(model, model_size, factor, warmup, betas=(0.9, 0.98), eps=1e-9):
    """
    获取Transformer优化器。
    参数：
    - model: 模型
    - model_size: 模型维度
    - factor: 缩放因子
    - warmup: 预热步数
    - betas: Adam优化器的beta参数
    - eps: Adam优化器的epsilon参数
    返回：
    - Transformer优化器实例
    """
    base = torch.optim.Adam(model.parameters(), betas=betas, eps=eps)
    return TransformerOptimizer(model_size, factor, warmup, base)
