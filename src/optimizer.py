import torch
import torch.optim as optim
import math
from torch.optim import Optimizer
from typing import Optional


class TransformerOptimizer:
    def __init__(self, model, model_size, factor=1.0, warmup=4000, optimizer=None):
        """
        初始化Transformer优化器。
        参数：
        - model: 模型参数
        - model_size: 模型维度
        - factor: 学习率缩放因子
        - warmup: 预热步数
        - optimizer: 基础优化器
        """
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.step_num = 0

        if optimizer is None:
            self.optimizer = optim.Adam(
                model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
            )
        else:
            self.optimizer = optimizer

        # 初始化优化器状态
        self.param_groups = self.optimizer.param_groups
        self._optimizer_step_pre_hooks = {}
        self._optimizer_state_dict_pre_hooks = {}
        self._optimizer_load_state_dict_pre_hooks = {}
        self._optimizer_state_dict_post_hooks = {}
        self._optimizer_load_state_dict_post_hooks = {}

    def step(self, closure=None):
        """
        执行单步优化。
        参数：
        - closure: 重新评估模型并返回损失的闭包
        返回：
        - 损失值
        """
        # 更新学习率
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.rate()

        # 执行步骤
        self.step_num += 1

        # 如果提供了闭包，调用优化器step方法并返回闭包结果
        if closure is not None:
            loss = self.optimizer.step(closure)
            return loss
        else:
            self.optimizer.step()
            return None

    def rate(self):
        """
        计算当前学习率。
        返回：
        - 学习率
        """
        # 在计算前确保step_num不为0
        step_num = max(1, self.step_num)
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step_num ** (-0.5), step_num * self.warmup ** (-1.5))
        )

    def zero_grad(self, set_to_none=False):
        """
        清除所有参数的梯度。
        参数：
        - set_to_none: 如果为True，将梯度设置为None而不是0
        """
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """
        返回优化器的状态字典。
        返回：
        - 状态字典
        """
        state_dict = {
            "step_num": self.step_num,
            "model_size": self.model_size,
            "factor": self.factor,
            "warmup": self.warmup,
            "optimizer": self.optimizer.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        """
        加载优化器状态。
        参数：
        - state_dict: 状态字典
        """
        self.step_num = state_dict["step_num"]
        self.model_size = state_dict["model_size"]
        self.factor = state_dict["factor"]
        self.warmup = state_dict["warmup"]
        self.optimizer.load_state_dict(state_dict["optimizer"])


def get_optimizer(model, model_size, factor=2.0, warmup=4000):
    """
    创建Transformer优化器。
    参数：
    - model: 模型
    - model_size: 模型维度
    - factor: 学习率缩放因子
    - warmup: 预热步数
    返回：
    - Transformer优化器
    """
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    return TransformerOptimizer(
        model=model,
        model_size=model_size,
        factor=factor,
        warmup=warmup,
        optimizer=optimizer,
    )
