"""
训练相关的工具函数和优化器实现


这个文件包含了训练Transformer模型所需的各种工具函数，包括：
- 损失函数（交叉熵）
- 优化器（AdamW）
- 学习率调度器（余弦退火）
- 梯度裁剪
- 数据批处理
- 模型检查点保存/加载


深度学习小知识：
   - 交叉熵损失：衡量模型预测分布与真实分布的差异
   - AdamW优化器：Adam的改进版本，更好地处理权重衰减
   - 学习率调度：动态调整学习率，提高训练效果
   - 梯度裁剪：防止梯度爆炸，提高训练稳定性
"""

import torch
import numpy as np
import math
import os
from typing import Optional, Callable, Iterable, BinaryIO, IO


def cross_entropy(o_i, y_i):
    """
    交叉熵损失函数（Cross Entropy Loss）


    深度学习小知识：
        - 交叉熵是分类任务中最常用的损失函数
        - 它衡量模型预测的概率分布与真实分布的差异
        - 公式：CE = -sum(y_true * log(y_pred))
        - 对于单标签分类，可以简化为：CE = -log(y_pred[true_class])


    【PyTorch替换说明】
    可以直接用 torch.nn.functional.cross_entropy 替换：
        import torch.nn.functional as F
        F.cross_entropy(o_i, y_i)


    实现区别：
        - 功能完全等价
        - PyTorch官方实现可能包含更多优化


    参数说明：
        o_i (torch.Tensor): 模型输出的logits，形状为 (batch_like, vocab_size)
        y_i (torch.Tensor): 真实标签，形状为 (batch_like,)


    返回：
        torch.Tensor: 平均交叉熵损失（标量）


    数值稳定性技巧：
        - 减去最大值：防止exp(x)溢出
        - keepdim=True：保持维度，避免广播错误
        - 合并log和exp：log(exp(x)) = x，但数值更稳定
    """
    # 为了数值稳定性，减去最大值
    # keepdim=True保持维度不变，方便后续广播
    o_i = o_i - o_i.max(dim=-1, keepdim=True).values

    # 获取真实标签对应的logit值
    # gather: 从o_i的最后一维中，以y_i为索引，取出对应的元素
    o_y = o_i.gather(dim=-1, index=y_i.unsqueeze(-1)).squeeze(-1)

    # 计算logsumexp：log(sum(exp(o_i)))
    # 这个计算方式比直接计算更数值稳定
    logsumexp = torch.log(torch.exp(o_i).sum(dim=-1))

    # 交叉熵 = -log(exp(o_y) / sum(exp(o_i))) = -o_y + logsumexp
    ce = -o_y + logsumexp  # (batch_like,)

    # 返回批次平均损失
    ce = ce.mean()  # scalar
    return ce


class AdamW(torch.optim.AdamW):
    """
    AdamW优化器（Adam with Decoupled Weight Decay）


    深度学习小知识：
        - Adam优化器：
          结合了动量（Momentum）和自适应学习率（Adaptive Learning Rate）
          - 动量：记录梯度的历史方向，加速收敛
          - 自适应学习率：根据梯度的二阶矩调整学习率
        - AdamW改进：
          将权重衰减（Weight Decay）与梯度更新解耦
          - Adam：权重衰减被加入到梯度中
          - AdamW：权重衰减独立应用于参数
          - AdamW在正则化方面表现更好


    【PyTorch替换说明】
    可以直接用 torch.optim.AdamW 替换：
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


    实现区别：
        - 功能完全等价
        - 这个实现展示了AdamW的内部工作原理


    参数说明：
        params: 可训练参数列表
        lr (float): 学习率
        betas (tuple): 动量参数，默认为(0.9, 0.999)
        eps (float): 数值稳定性的小常数，默认为1e-8
        weight_decay (float): 权重衰减系数，默认为0
    """

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        执行一步参数更新


        深度学习小知识：
            - @torch.no_grad()装饰器：
              表示这个操作不需要计算梯度，可以节省内存
            - 闭包（closure）：
              用于支持某些优化器的两次梯度计算功能


        参数说明：
            closure (Callable, optional): 可选的闭包函数


        返回：
            loss (torch.Tensor, optional): 如果提供了闭包，返回损失值
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():  # 闭包通常需要重新计算梯度
                loss = closure()

        # 遍历所有参数组
        for group in self.param_groups:
            alpha = group['lr']  # 学习率
            beta1, beta2 = group['betas']  # 动量参数
            eps = group['eps']  # 数值稳定性常数
            lambda_ = group['weight_decay']  # 权重衰减系数

            # 遍历每个参数
            for theta in group['params']:
                if theta.grad is None:
                    continue

                grad = theta.grad.data
                state = self.state[theta]

                # 状态初始化：为每个参数维护一阶矩和二阶矩
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(theta.data)  # 一阶矩（动量）
                    state['v'] = torch.zeros_like(theta.data)  # 二阶矩（自适应学习率）

                m, v = state['m'], state['v']
                state['step'] += 1  # 步数从1开始
                t = state['step']

                # 更新一阶矩：m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                # 使用原地操作（mul_、add_）节省内存
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新二阶矩：v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算偏差修正后的学习率
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                alpha_t = alpha * math.sqrt(bias_correction2) / bias_correction1

                # 更新参数：theta = theta - alpha_t * m / (sqrt(v) + eps)
                denom = v.sqrt().add_(eps)
                theta.addcdiv_(m, denom, value=-alpha_t)

                # 应用权重衰减（AdamW的核心）
                # 权重衰减独立于梯度更新
                if lambda_ != 0:
                    theta.mul_(1 - alpha * lambda_)

        return loss


def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    """
    余弦退火学习率调度器（Cosine Annealing Learning Rate Schedule）


    深度学习小知识：
        - 学习率调度：
          动态调整学习率，提高训练效果
        - 预热（Warm-up）：
          在训练开始时，学习率从0逐渐增加到最大值
          好处：稳定训练初期的梯度更新
        - 余弦退火：
          学习率按照余弦函数从最大值衰减到最小值
          好处：平滑的衰减，避免学习率突然下降


    【PyTorch替换说明】
    可以用 torch.optim.lr_scheduler.CosineAnnealingLR 替换：
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_c-T_w, eta_min=alpha_min)


    实现区别：
        - 功能类似，但这个实现包含了预热阶段
        - PyTorch官方实现需要额外处理预热


    参数说明：
        t (int): 当前步数
        alpha_max (float): 最大学习率
        alpha_min (float): 最小（最终）学习率
        T_w (int): 预热步数
        T_c (int): 余弦退火的总步数


    返回：
        float: 当前步数的学习率


    学习率曲线：
        - [0, T_w): 线性增长
        - [T_w, T_c]: 余弦衰减
        - (T_c, ∞): 保持最小值
    """
    # 预热阶段：线性增长
    if t < T_w:
        alpha_t = alpha_max * t / T_w
    # 余弦退火阶段：从最大值衰减到最小值
    elif t >= T_w and t <= T_c:
        # 计算余弦函数的角度
        temp = math.pi * (t - T_w) / (T_c - T_w)
        # 余弦退火公式
        alpha_t = alpha_min + 1 / 2 * (1 + math.cos(temp)) * (alpha_max - alpha_min)
    # 保持最小值阶段
    elif t > T_c:
        alpha_t = alpha_min

    return alpha_t


def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    """
    梯度裁剪（Gradient Clipping）


    深度学习小知识：
        - 梯度爆炸问题：
          在深度网络中，梯度可能会变得非常大，导致训练不稳定
        - 梯度裁剪的作用：
          当梯度范数超过阈值时，将梯度缩放到阈值范围内
        - 好处：
          1. 防止梯度爆炸
          2. 提高训练稳定性
          3. 允许使用更大的学习率


    【PyTorch替换说明】
    可以用 torch.nn.utils.clip_grad_norm_ 替换：
        torch.nn.utils.clip_grad_norm_(params, max_norm)


    实现区别：
        - 功能完全等价
        - PyTorch官方实现可能包含更多优化


    参数说明：
        params (Iterable[torch.nn.Parameter]): 可训练参数列表
        max_norm (float): 最大梯度范数
        eps (float): 数值稳定性的小常数，默认为1e-6


    返回：
        float: 裁剪前的梯度范数（用于监控）
    """
    params = [p for p in params if p.grad is not None]
    if len(params) == 0:
        return 0.0

    total_sq_norm = None
    for p in params:
        grad = p.grad.detach()
        grad_sq_norm = torch.sum(grad * grad)
        total_sq_norm = grad_sq_norm if total_sq_norm is None else total_sq_norm + grad_sq_norm

    g_norm = torch.sqrt(total_sq_norm)

    if g_norm >= max_norm:
        clip_coef = max_norm / (g_norm + eps)
        for p in params:
            p.grad.detach().mul_(clip_coef)

    return g_norm  # 返回裁剪前的梯度范数（用于监控）


def get_batch(data, batch_size, context_length, device):
    """
    获取训练批次数据


    深度学习小知识：
        - 批处理（Batch Processing）：
          一次性处理多个样本，而不是单个样本
          好处：
          1. GPU利用率更高
          2. 梯度估计更稳定
        - 上下文长度（Context Length）：
          模型一次能够处理的序列长度
          例如：预测下一个词时，模型会考虑前面N个词
        - np.memmap：
          内存映射文件，可以处理比RAM大的文件
          数据按需从磁盘加载到内存，节省内存


    参数说明：
        data: 输入序列数据，可以是numpy数组或np.memmap对象
        batch_size (int): 批次大小
        context_length (int): 上下文长度
        device: 计算设备（CPU或GPU）


    返回：
        x (torch.Tensor): 输入token ID，形状为 (batch_size, context_length)
        y (torch.Tensor): 目标token ID（即x的下一个token），形状为 (batch_size, context_length)


    注意：
        当前的实现是纯随机采样，可能导致一个Epoch内数据重复或遗漏。
        进阶技巧：可以先生成所有可能的索引，然后打乱并按顺序取。


    Batch（批次）
    定义：一次训练迭代中使用的样本子集
    原因：内存限制，无法一次性加载所有数据
    Batch Size：每个批次包含的样本数量
    作用：通过小批量数据计算梯度，更新模型参数
    Epoch（轮次）
    定义：完整遍历整个训练数据集一次
    作用：让模型多次学习相同数据，提高泛化能力
    关系：1 Epoch = 所有 Batch 的总和


    总迭代次数 = (样本总数 ÷ Batch Size) × Epoch 数
    ## 示例
    假设有 1000 个样本，Batch Size = 100，Epoch = 3：
    - 每个 Epoch 有 10 个 Batch
    - 总共训练 30 次迭代
    - 模型会看到 3 次完整数据集
    """
    ix = np.random.randint(0, len(data) - context_length, size=batch_size)
    x_np = np.empty((batch_size, context_length), dtype=np.int64)
    y_np = np.empty((batch_size, context_length), dtype=np.int64)

    for row, start in enumerate(ix):
        x_np[row] = data[start: start + context_length]
        y_np[row] = data[start + 1: start + context_length + 1]

    x = torch.from_numpy(x_np).to(device)
    y = torch.from_numpy(y_np).to(device)

    return x, y


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int,
                    out: str | os.PathLike | BinaryIO | IO[bytes]):
    """
    保存模型检查点


    深度学习小知识：
        - 检查点（Checkpoint）：
          保存训练过程中的模型状态，包括：
          1. 模型参数
          2. 优化器状态
          3. 训练进度（迭代次数）
        - 作用：
          1. 从中间状态恢复训练
          2. 选择最佳模型
          3. 防止训练意外中断


    参数说明：
        model (torch.nn.Module): 要保存的模型
        optimizer (torch.optim.Optimizer): 优化器
        iteration (int): 当前迭代次数
        out: 输出路径或文件对象


    【PyTorch替换说明】
    可以使用 torch.save 直接保存，但需要手动组织数据结构。
    """
    obj = {
        'model': model.state_dict(),  # 模型参数
        'optimizer': optimizer.state_dict(),  # 优化器状态
        'iteration': iteration,  # 训练进度
    }
    torch.save(obj, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    加载模型检查点


    参数说明：
        src: 检查点文件路径或文件对象
        model (torch.nn.Module): 要加载参数的模型
        optimizer (torch.optim.Optimizer): 要加载状态的优化器


    返回：
        int: 保存时的迭代次数


    【PyTorch替换说明】
    可以使用 torch.load 直接加载，然后手动加载到模型和优化器。
    """
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iteration']