"""
从零实现的Transformer模型核心组件


本文件实现了Transformer架构的所有关键组件，完全不依赖torch.nn的高层抽象（如nn.Linear、nn.LayerNorm等）。
这样做是为了深入理解Transformer的底层工作原理。


深度学习小知识：
- Transformer是一种基于注意力机制的神经网络架构，主要用于处理序列数据（如文本）
- 它通过"自注意力"机制让模型能够理解序列中不同位置之间的关系
- 相比之前的RNN/LSTM，Transformer可以并行处理整个序列，训练效率更高
"""

import torch
from torch import nn
import math


class Linear(nn.Module):
    """
    自定义线性层（全连接层）


    线性层是神经网络中最基础的组件，它执行矩阵乘法：y = xW^T + b
    这个实现不使用偏置项（bias），只包含权重矩阵。


    深度学习小知识：
    - 线性层的作用是对输入数据进行线性变换，将数据从一个维度映射到另一个维度
    - 例如：输入是512维的向量，通过线性层可以映射到2048维的向量
    - 这是神经网络"学习"的基础，通过调整权重矩阵W来适应不同的任务


    什么是偏置项（bias）
    - 想象一下，你每天早上决定要不要穿外套。这个决定可能取决于：
        温度（输入数据）
        你的个人习惯（比如你特别怕冷，或者特别耐热）
        这里的"温度"就是神经网络的输入，而"你的个人习惯"就是偏置项。
        具体例子，假设有一个简单的判断：
        温度 ≥ 20℃ → 不穿外套
        温度 < 20℃ → 穿外套
        但如果一个人特别怕冷，他可能 15℃ 就觉得冷了；另一个人特别耐热，可能 10℃ 才穿外套。
        偏置项就是调整这个"门槛"的值！


        数学上的理解
        神经网络的每个神经元会做这样的计算：
        输出 = 输入 × 权重 + 偏置
        权重：决定输入有多重要（比如温度对决定的影响程度）
        偏置：调整激活的"门槛"（就像调整温度计的刻度）
    - 为什么需要偏置？
        如果没有偏置，所有神经元的决策线都必须通过原点（0,0），这就像规定"温度为0时必须穿外套"，太死板了！
        有了偏置，决策线可以平移，更灵活地适应各种情况。


    【PyTorch替换说明】
    可以直接用 torch.nn.Linear 替换：
        self.linear = nn.Linear(in_features, out_features, bias=False)


    实现区别：
    1. PyTorch官方nn.Linear的权重形状是 (out_features, in_features)
       我们的实现也是 (out_features, in_features)，但forward时需要转置
    2. PyTorch官方实现支持偏置项（bias），我们的实现不支持
    3. PyTorch官方默认使用Kaiming初始化，我们使用截断正态分布初始化


    使用Kaiming做线性层初始化，和使用截断正态分布初始化有什么区别
    - 1. 什么是“初始化”？为什么需要它？
        想象一下，你有一群新同学（神经网络中的“神经元”），他们要一起学习解决一个问题。在他们开始学习之前，你需要给他们一些初始的“想法”或“观点”（也就是神经网络中的“权重”和“偏置”）。


        如果初始想法太离谱：比如一开始就让他们觉得所有东西都是错的，或者所有东西都是对的，那他们就很难从头开始学好。
        如果初始想法太接近：所有同学的想法都一模一样，那他们也学不到什么新东西。
        所以，初始化就是给神经网络的“权重”和“偏置”设置一个合适的起始值。一个好的初始化能让神经网络更快、更稳定地学到东西，避免在学习过程中出现“卡住”或者“跑偏”的情况。


    - 2. 截断正态分布初始化（Truncated Normal Distribution Initialization）
        怎么理解？ 想象你有一堆铅笔，它们的长度是随机的，但大多数铅笔的长度都集中在平均值附近，只有少数特别长或特别短的。这就是“正态分布”。 “截断”的意思是，如果你随机抽到一支铅笔，它太长或太短了，我们就把它扔掉，重新抽，直到抽到的铅笔长度在某个合理的范围内。


        作用： 这种初始化方法就是从一个“正态分布”中随机选取权重值，但会把那些离平均值太远（也就是太大或太小）的权重值“截断”掉，重新选取，直到它们都在一个合理的范围内。 这样做的目的是为了确保初始权重不会太大（导致学习过程“爆炸”），也不会太小（导致学习过程“停滞”），给神经网络一个比较“温和”的开端。它是一种比较通用的初始化方法。


    - 3. Kaiming初始化（Kaiming Initialization）
        怎么理解？ Kaiming初始化更像是一种“针对性”的初始化方法，它主要为一种特殊的“同学”（叫做ReLU激活函数）设计。


        什么是ReLU激活函数？ 我们之前提到“偏置项”就像调整“门槛”。ReLU激活函数就像一个“只说好话”的同学：


        如果输入是正数，它就输出这个正数。
        如果输入是负数，它就直接输出0（就像没听到一样）。
        Kaiming初始化有什么特别的？ 因为ReLU激活函数会把一半的负数输入变成0，这意味着很多信息可能会“消失”或“被忽略”。如果网络很深，经过很多层ReLU，信号可能会越来越弱，导致学习困难。


        Kaiming初始化就是为了解决这个问题。它会根据神经元的输入连接数量来智能地设置初始权重，确保即使有一半的神经元被ReLU“关闭”了，剩下的活跃神经元传递的信号强度也能保持在一个稳定的水平。


        作用： Kaiming初始化能让信号在深度神经网络中流动时，既不会越来越弱（信息丢失），也不会越来越强（信息爆炸），从而让深度网络（特别是使用ReLU的）能够更有效地学习。


    截断正态分布就像给你一个通用的、比较稳妥的开始，不至于太好也不至于太坏。
    Kaiming就像是给使用ReLU的同学量身定制的开始，能让他们在学习过程中发挥出最佳状态，特别是在学习内容很多、很深（网络很深）的时候。
    现在大多数现代神经网络都使用ReLU作为激活函数，所以Kaiming初始化（或其变体）是非常常用且有效的选择




    参数说明：
        in_features (int): 输入特征维度，即输入向量的长度
        out_features (int): 输出特征维度，即输出向量的长度
        device: 计算设备（CPU或GPU）
        dtype: 数据类型（如torch.float32）
    """

    # todo torch.nn.Linear如何修改为截断正态分布初始化？

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features  # 输入维度
        self.out_features = out_features  # 输出维度

        # 初始化权重矩阵，形状为 (out_features, in_features)
        # 注意：这里使用行优先（row-major）的内存布局，这是PyTorch的默认方式
        # 权重矩阵的形状为什么是 (out_features, in_features) 而不是 (in_features, out_features)？
        # 这是因为在forward中我们使用 x @ self.weight.T，即 x 乘以 权重的转置
        # 这样做是为了与PyTorch的内存布局保持一致
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        # 使用截断正态分布初始化权重
        # 深度学习小知识：权重初始化非常重要，好的初始化可以让模型更快收敛
        # 截断正态分布：在正态分布的基础上，截断超出[-3*sigma, 3*sigma]范围的值
        # sigma（标准差）的计算公式：sqrt(2 / (in_features + out_features))
        # 这个公式来自Xavier初始化（也叫Glorot初始化），专门用于线性层
        sigma = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数


        参数说明：
            x (torch.Tensor): 输入张量，形状可以是任意维度，但最后一个维度必须是 in_features
                             例如：(batch_size, seq_len, in_features) 或 (batch_size, in_features)


        返回：
            torch.Tensor: 输出张量，形状与输入相同，但最后一个维度变为 out_features
                         例如：(batch_size, seq_len, out_features) 或 (batch_size, out_features)


        数学原理：
            y = x @ W^T
            其中 @ 表示矩阵乘法，W^T 表示权重矩阵的转置


        为什么需要转置？
        - 假设 x 的形状是 (B, T, C)，其中 C = in_features
        - weight 的形状是 (out_features, in_features) = (C', C)
        - 我们想要的结果是 (B, T, C')
        - 直接相乘 x @ weight.T 会得到 (B, T, C) @ (C', C)^T = (B, T, C')
        - 所以我们需要 x @ weight.T，即 x 乘以 weight 的转置
        """
        # x @ self.weight.T 等价于 torch.matmul(x, self.weight.T)
        # @ 是Python 3.5+引入的矩阵乘法运算符
        return x @ self.weight.T


class Embedding(nn.Module):
    """
    词嵌入层（Embedding Layer）


    词嵌入层将离散的token ID转换为连续的向量表示。


    深度学习小知识：
    - 计算机无法直接理解文本，需要将文本转换为数字
    - 最简单的方法是给每个词分配一个唯一的ID（如"猫"=1001，"狗"=1002）
    - 但ID之间没有语义关系（1001和1002只是数字，不代表"猫"和"狗"相似）
    - 词嵌入将每个ID映射到一个高维向量（如512维），在这个向量空间中，语义相似的词距离更近
    - 例如："猫"和"狗"的向量距离可能很近，而"猫"和"汽车"的向量距离很远


    【PyTorch替换说明】
    可以直接用 torch.nn.Embedding 替换：
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)


    实现区别：
    1. PyTorch官方nn.Embedding默认使用正态分布初始化，标准差为1
       我们的实现使用截断正态分布，标准差为1，截断范围[-3, 3]
    2. 功能上完全等价


    参数说明：
        num_embeddings (int): 词表大小，即模型能识别的不同token的总数
                             例如：如果词表有10000个词，那么num_embeddings=10000
        embedding_dim (int): 嵌入向量的维度，即每个token被表示为多长的向量
                             例如：512表示每个token用一个512维的向量表示
    """

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.vocab_size = num_embeddings  # 词表大小
        self.d_model = embedding_dim  # 嵌入向量的维度

        # 初始化嵌入矩阵，形状为 (vocab_size, d_model)
        # 这个矩阵就像一个"字典"，每一行对应一个token的向量表示
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model, device=device, dtype=dtype))

        # 使用截断正态分布初始化嵌入矩阵
        # 标准差为1，截断范围[-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数：根据token ID查找对应的嵌入向量


        参数说明：
            token_ids (torch.Tensor): token ID的张量，形状任意
                                     例如：(batch_size, seq_len) 表示一个批次中每个token的ID


        返回：
            torch.Tensor: 嵌入向量，形状与输入相同，但最后一个维度变为 d_model
                         例如：(batch_size, seq_len, d_model)


        工作原理：
            - 假设 token_ids = [[1001, 1002, 1003], [1004, 1005, 1006]] 形状为 (2, 3)
            - self.weight[token_ids] 会从嵌入矩阵中取出第1001、1002、1003、1004、1005、1006行
            - 返回形状为 (2, 3, d_model) 的张量


        深度学习小知识：
            - 这个操作本质上是一个"查表"操作
            - 在训练过程中，嵌入矩阵的值会不断更新，使得语义相似的token的向量距离更近
        """
        # 使用索引操作从嵌入矩阵中查找对应的向量
        # 这就像查字典：输入"猫"的ID，输出"猫"的向量表示
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    RMSNorm（Root Mean Square Layer Normalization）归一化层


    RMSNorm是一种比LayerNorm更高效的归一化方法，被Llama等现代大模型广泛使用。


    深度学习小知识：
    - 归一化（Normalization）是深度学习中非常重要的技术
    - 它的作用是将数据缩放到一个标准范围内，防止数值过大或过小
    - 归一化可以让模型训练更稳定、收敛更快
    - LayerNorm会对每个样本的所有特征进行归一化，计算均值和标准差
    - RMSNorm只计算均方根（Root Mean Square），不计算均值，计算量更小


    RMSNorm vs LayerNorm：
    - LayerNorm: (x - mean) / std，其中std = sqrt(mean((x - mean)^2) + eps)
    - RMSNorm: x / rms，其中rms = sqrt(mean(x^2) + eps)
    - RMSNorm省去了计算均值和减去均值的步骤，速度更快


    【PyTorch替换说明】
    不能直接用 torch.nn.LayerNorm 替换，因为LayerNorm的实现方式不同。
    但可以使用自定义实现或第三方库（如transformers）的RMSNorm。


    实现区别：
    - PyTorch官方nn.LayerNorm计算均值和标准差，RMSNorm只计算均方根
    - RMSNorm的计算量约为LayerNorm的一半


    参数说明：
        d_model (int): 特征维度，即输入张量的最后一个维度的大小
        eps (float): 一个很小的数值（如1e-5），用于防止除以0
                     当计算rms时，如果x全为0，则sqrt(0) = 0，加上eps可以避免除以0
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model  # 特征维度
        self.eps = eps  # 防止除以0的小常数

        # gain是一个可学习的参数，初始值为全1
        # 形状为 (d_model,)
        # gain的作用是让模型可以学习到"缩放"的程度，而不是强制归一化到固定范围
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数：对输入进行RMSNorm归一化


        参数说明：
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)
                             简记为 (B, T, C)，其中C = d_model


        返回：
            torch.Tensor: 归一化后的张量，形状与输入相同 (B, T, C)


        计算步骤：
            1. 将输入转换为float32类型（提高数值精度）
            2. 计算均方根：rms = sqrt(mean(x^2) + eps)
            3. 归一化：x_normalized = x / rms
            4. 乘以可学习的gain：output = gain * x_normalized
            5. 转换回原始数据类型


        数学公式：
            output = gain * (x / sqrt(mean(x^2) + eps))


        深度学习小知识：
            - 为什么要在计算rms时转换为float32？
              因为float16或bfloat16的精度较低，计算平方和开方时可能会出现数值不稳定
            - keepdim=True的作用是什么？
              保持维度不变，例如从 (B, T, C) 计算均值后得到 (B, T, 1) 而不是 (B, T)
              这样方便后续的广播操作
        """
        # 保存输入的数据类型
        in_dtype = x.dtype

        # 转换为float32进行计算（提高数值稳定性）
        x = x.to(torch.float32)

        # 计算均方根（Root Mean Square）
        # 步骤1：计算x的平方：x^2
        # 步骤2：在最后一个维度上求均值：mean(x^2, dim=-1)
        # 步骤3：加上eps防止除以0：mean(x^2) + eps
        # 步骤4：开平方根：sqrt(...)
        # keepdim=True保持维度不变，从 (B, T, C) 变为 (B, T, 1)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # 归一化：x除以rms
        # 广播机制：rms的形状是 (B, T, 1)，x的形状是 (B, T, C)
        # PyTorch会自动将rms广播到 (B, T, C)
        x = x / rms

        # 转换回原始数据类型
        x = x.to(in_dtype)

        # 乘以可学习的gain
        # gain的形状是 (C,)，会被广播到 (1, 1, C)，然后与x相乘
        # 这样每个特征维度都有一个独立的缩放因子
        return self.gain * x


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU激活函数（Sigmoid Linear Unit）


    SiLU激活函数是Sigmoid函数的加权版本，也被称为Swish激活函数。


    数学公式：
        silu(x) = x * sigmoid(x)


    深度学习小知识：
        - 激活函数的作用是引入非线性，让神经网络能够学习复杂的模式
        - 如果没有激活函数，多层线性层的组合仍然等价于单层线性层
        - SiLU/Swish在某些情况下比ReLU表现更好，特别是在深度网络中


    【PyTorch替换说明】
    可以直接用 torch.nn.functional.silu 替换：
        import torch.nn.functional as F
        F.silu(x)


    实现区别：
        - 功能完全等价
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU前馈网络（Feed-Forward Network）


    SwiGLU是Transformer中前馈网络的一种变体，结合了SiLU激活函数和门控线性单元（GLU）。
    被Llama、PaLM等现代大模型广泛使用。


    深度学习小知识：
        - Transformer中的前馈网络作用：对每个位置的表示进行非线性变换
        - 传统Transformer使用ReLU激活函数：FFN(x) = ReLU(xW1 + b1)W2 + b2
        - SwiGLU使用门控机制：FFN(x) = (SiLU(xW1) ⊙ (xW3))W2
        - ⊙表示逐元素相乘（Hadamard积）
        - 门控机制可以让模型学习到"哪些信息应该被传递"


    结构说明：
        - w1: 将输入从d_model维度映射到d_ff维度（门控分支）
        - w3: 将输入从d_model维度映射到d_ff维度（值分支）
        - w2: 将d_ff维度映射回d_model维度（输出投影）


    参数说明：
        d_model (int): 输入和输出的特征维度
        d_ff (int): 前馈网络中间层的维度，通常比d_model大很多
                    例如：d_model=512，d_ff=2048
    """

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # w1: 门控分支的线性变换（d_model -> d_ff）
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)

        # w3: 值分支的线性变换（d_model -> d_ff）
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        # w2: 输出投影（d_ff -> d_model）
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数：SwiGLU计算过程


        参数说明：
            x (torch.Tensor): 输入张量，形状为 (B, T, C)


        返回：
            torch.Tensor: 输出张量，形状与输入相同 (B, T, C)


        计算步骤：
            1. x1 = w1(x)   # 门控分支，形状变为 (B, T, d_ff)
            2. x3 = w3(x)   # 值分支，形状变为 (B, T, d_ff)
            3. gate = SiLU(x1)  # 对门控分支应用SiLU激活函数
            4. gated_value = gate * x3  # 门控机制：门控信号乘以值
            5. output = w2(gated_value)  # 投影回原始维度


        数学公式：
            output = w2( SiLU(w1(x)) ⊙ w3(x) )


        深度学习小知识：
            - 门控机制的作用：让模型学习到"哪些信息应该被传递"
            - SiLU(x1)的输出在0到1之间，就像一个"开关"
            - 当SiLU(x1)接近1时，x3的信息完全传递
            - 当SiLU(x1)接近0时，x3的信息被抑制
            - 这种机制让模型能够选择性地传递信息
        """
        # x的形状: (batch_size, seq_len, d_model) = (B, T, C)

        # 门控分支：将输入映射到高维空间
        x1 = self.w1(x)  # (B, T, d_ff)

        # 值分支：将输入映射到高维空间
        x3 = self.w3(x)  # (B, T, d_ff)

        # 门控机制：SiLU(x1) * x3
        # SiLU(x1)产生一个0到1之间的门控信号
        # 然后与x3逐元素相乘
        x = silu(x1) * x3  # (B, T, d_ff)

        # 投影回原始维度
        x = self.w2(x)  # (B, T, d_model)

        return x


class RoPE_llama(nn.Module):
    """
    RoPE（Rotary Position Embeddings）旋转位置编码 - Llama风格


    RoPE是一种先进的位置编码方法，通过旋转矩阵将位置信息编码到query和key向量中。


    深度学习小知识：
        - 位置编码的作用：让模型知道序列中每个token的位置信息
        - 传统方法：在输入中加入固定的位置编码向量
        - RoPE方法：通过旋转query和key向量来编码位置信息
        - 优点：相对位置编码，可以处理任意长度的序列


    数学原理：
        - 将d_k维的向量分成d_k/2个2维向量对
        - 对每个2维向量对进行旋转，旋转角度与位置相关
        - 旋转矩阵：[cos(mθ) -sin(mθ); sin(mθ) cos(mθ)]
        - 其中m是位置，θ是频率参数


    【PyTorch替换说明】
    不能直接用PyTorch内置函数替换，RoPE是相对较新的技术。
    但可以使用transformers库中的RotaryEmbedding类。


    参数说明：
        theta (float): 频率参数，控制旋转的频率，通常为10000.0
        d_k (int): query/key向量的维度，必须是偶数
        max_seq_len (int): 预计算的最大序列长度
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires d_k to be even."
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        self.device = device

        # 计算频率：f_i = 1 / (theta^(2i/d_k))，其中i=0,1,...,d_k/2-1
        # 这些频率用于生成旋转角度
        p = torch.arange(0, d_k // 2, dtype=torch.float64, device=device)
        inv_freq = 1.0 / (self.theta ** (2.0 * p / d_k))

        # 转换为float32（提高兼容性，特别是MPS设备）
        inv_freq = inv_freq.to(torch.float32)

        # 预计算所有位置的角度：角度 = 位置 * 频率
        # positions: 0, 1, 2, ..., max_seq_len-1
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (L,)
        # angles[i, j] = positions[i] * inv_freq[j]
        angles = torch.einsum("i,j->ij", positions, inv_freq)  # (L, d_k/2)

        # 预计算并缓存cos和sin值
        # persistent=False表示不保存到checkpoint中（可以随时重新计算）
        self.register_buffer("cos_cached", angles.cos(), persistent=False)  # (L, d_k/2)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)  # (L, d_k/2)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        应用旋转位置编码的核心函数


        参数说明：
            x (torch.Tensor): 输入张量，形状为 (..., seq_len, d_k)
            cos (torch.Tensor): cos值，形状为 (..., seq_len, d_k/2)
            sin (torch.Tensor): sin值，形状为 (..., seq_len, d_k/2)


        返回：
            torch.Tensor: 旋转后的张量，形状与输入相同 (..., seq_len, d_k)


        数学原理：
            将d_k维向量分成d_k/2个2维向量对
            对每个2维向量对应用旋转矩阵：
                [x_even'] = x_even*cos - x_odd*sin
                [x_odd' ] = x_even*sin + x_odd*cos


        深度学习小知识：
            - 这种旋转操作保持了向量的模长不变（旋转是正交变换）
            - 不同位置的向量旋转角度不同，从而编码了位置信息
            - 相对位置关系通过旋转角度的差值来体现
        """
        # 将向量分成偶数和奇数部分
        # x[..., 0::2]: 从第0个元素开始，每隔2个取一个（偶数索引）
        # x[..., 1::2]: 从第1个元素开始，每隔2个取一个（奇数索引）
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k/2)

        # 应用旋转矩阵
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # 将旋转后的偶数和奇数部分交错放回
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数


        参数说明：
            x (torch.Tensor): 输入张量，形状为 (..., seq_len, d_k)
            token_positions (torch.Tensor): token位置信息，形状可以是：
                - (seq_len,): 单个序列的位置
                - (batch, seq_len): 批次中每个序列的位置
                - None: 自动生成0到seq_len-1的位置


        返回：
            torch.Tensor: 应用RoPE后的张量，形状与输入相同


        处理步骤：
            1. 验证输入维度
            2. 处理位置信息（转换为long类型）
            3. 检查位置是否超出预计算范围
            4. 从缓存中获取对应的cos/sin值
            5. 应用旋转操作
        """
        assert x.size(-1) == self.d_k, f"Expected last dim d_k={self.d_k}, got {x.size(-1)}"
        seq_len = x.size(-2)

        # 处理位置信息
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            token_positions = token_positions.to(device=x.device, dtype=torch.long)

        # 检查最大位置是否超出预计算范围
        # torch.numel()返回张量中元素的总数
        max_pos = int(token_positions.max().item()) if token_positions.numel() > 0 else 0
        if max_pos >= self.max_seq_len:
            raise ValueError(
                f"token_positions has max={max_pos}, but max_seq_len={self.max_seq_len}. "
                "Please increase max_seq_len in RoPE init."
            )

        # 从缓存中获取对应的cos/sin值
        # index_select: 根据索引从缓存中选取值
        cos = self.cos_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)
        sin = self.sin_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)

        # 检查序列长度对齐
        if cos.shape[-2] != seq_len:
            raise ValueError(f"token_positions seq dim {cos.shape[-2]} != x seq_len {seq_len}")

        # 数据类型对齐
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        return self._apply_rope(x, cos, sin)


'''
Q：RoPE（Rotary Position Embeddings）旋转位置编码 - Llama风格和惰性/自动扩展的RoPE的区别是什么？那个更好


A：
核心区别：什么时候准备好"位置信息"
想象你在准备一场考试，需要准备1000道题的答案：


Llama风格（RoPE_llama）
特点：提前准备好所有答案


初始化时：预计算 0~max_seq_len-1 位置的 cos/sin 值
运行时：直接从缓存中查找
```


**比喻：** 就像考前把1000道题的答案都背好了，考试时直接回忆。


**优点：**
- 运行时速度快（直接查表）
- 实现简单直接


**缺点：**
- 必须提前知道最大序列长度
- 如果序列超过预设长度，直接报错


---


### 惰性/自动扩展（RoPE）


**特点：用到多少准备多少**


```
初始化时：不预计算任何东西
运行时：根据实际需要的位置，动态扩展缓存


比喻： 就像考试时遇到哪道题再准备哪道题的答案，用多少背多少。


优点：


不需要提前知道最大序列长度
更灵活，可以处理任意长度的序列
缺点：


首次遇到新位置时需要计算（有微小开销）
实现稍复杂
|特性 |Llama风格| 惰性/自动扩展 |
|初始化参数  |需要 max_seq_len    |不需要|
|缓存时机   |初始化时全部预计算 |运行时按需计算|
|超出范围   |直接报错  |自动扩展|
|内存占用   |固定（可能浪费）  |按需增长|


哪个更好？
取决于你的场景：


如果你知道最大序列长度（比如训练时固定长度2048）


Llama风格更简单，性能略好
如果你不确定序列长度（比如推理时可能处理超长文本）


惰性扩展更灵活，不会报错
实际使用中


两者性能差异很小（惰性扩展只在首次遇到新位置时有开销）
惰性扩展更通用
'''


class RoPE(nn.Module):
    """
    惰性/自动扩展的RoPE实现


    这是RoPE的另一种实现方式，不需要在初始化时指定最大序列长度。
    在forward时根据实际需要的位置自动扩展缓存。


    深度学习小知识：
        - 相比RoPE_llama，这种实现更加灵活
        - 不需要预先知道最大序列长度，可以处理任意长度的序列
        - 但每次遇到新的位置时都需要重新计算cos/sin值


    实现特点：
        - 初始化时不预计算任何cos/sin值
        - 只保存频率参数inv_freq
        - 在forward时根据需要的位置动态扩展缓存


    【PyTorch替换说明】
    不能直接用PyTorch内置函数替换，这是自定义实现。


    参数说明：
        theta (float): 频率参数，控制旋转的频率，通常为10000.0
        d_k (int): query/key向量的维度，必须是偶数
    """

    def __init__(self, theta: float, d_k: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires d_k to be even."
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.device = device

        # 计算频率参数
        p = torch.arange(0, d_k // 2, dtype=torch.float64, device=device)
        inv_freq = 1.0 / (self.theta ** (2.0 * p / d_k))

        # 注册频率参数（不保存到checkpoint）
        self.register_buffer("inv_freq", inv_freq.to(torch.float32), persistent=False)  # (d_k/2,)

        # 初始化空的cos/sin缓存
        # 在forward时会根据需要动态扩展
        self.register_buffer("cos_cached", torch.empty(0, d_k // 2, dtype=torch.float32, device=device),
                             persistent=False)
        self.register_buffer("sin_cached", torch.empty(0, d_k // 2, dtype=torch.float32, device=device),
                             persistent=False)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        应用旋转位置编码的核心函数


        参数说明：
            x (torch.Tensor): 输入张量，形状为 (..., seq_len, d_k)
            cos (torch.Tensor): cos值，形状为 (..., seq_len, d_k/2)
            sin (torch.Tensor): sin值，形状为 (..., seq_len, d_k/2)


        返回：
            torch.Tensor: 旋转后的张量，形状与输入相同 (..., seq_len, d_k)


        数学原理：
            将d_k维向量分成d_k/2个2维向量对
            对每个2维向量对应用旋转矩阵：
                [x_even'] = x_even*cos - x_odd*sin
                [x_odd' ] = x_even*sin + x_odd*cos


        深度学习小知识：
            - 这种旋转操作保持了向量的模长不变（旋转是正交变换）
            - 不同位置的向量旋转角度不同，从而编码了位置信息
            - 相对位置关系通过旋转角度的差值来体现
        """
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd = x[..., 1::2]  # (..., seq_len, d_k/2)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        # Interleave back (..., seq_len, d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    @torch.no_grad()
    def _maybe_extend_cache(self, needed_len: int, device: torch.device):
        """
        扩展cos/sin缓存以覆盖所需的位置范围


        参数说明：
            needed_len (int): 需要覆盖的最大位置（不包含）
            device (torch.device): 计算设备


        工作原理：
            - 检查当前缓存长度是否足够
            - 如果不够，计算新位置的cos/sin值
            - 将新值追加到现有缓存中


        深度学习小知识：
            - @torch.no_grad()装饰器表示这个操作不需要计算梯度
            - 这可以节省内存，因为不需要为这些缓存值存储梯度信息
            - 惰性计算：只在需要时才计算，避免浪费计算资源
        """
        cur_len = int(self.cos_cached.size(0))
        if needed_len <= cur_len:
            return

        # 计算需要扩展的位置范围 [cur_len, needed_len)
        new_positions = torch.arange(cur_len, needed_len, dtype=torch.float32, device=device)  # (ΔL,)

        # 计算新位置的旋转角度
        # angles[i, j] = new_positions[i] * inv_freq[j]
        inv_freq = self.inv_freq.to(device=device)
        angles = torch.einsum("i,j->ij", new_positions, inv_freq)  # (ΔL, d_k/2)

        # 计算cos和sin值
        new_cos = angles.cos().to(dtype=torch.float32)  # (ΔL, d_k/2)
        new_sin = angles.sin().to(dtype=torch.float32)

        # 如果缓存设备不正确，迁移到正确的设备
        # 例如：模块在CPU上创建，但输入在CUDA上
        if self.cos_cached.device != device:
            self.cos_cached = self.cos_cached.to(device=device)
            self.sin_cached = self.sin_cached.to(device=device)

        # 将新值追加到缓存中
        if cur_len == 0:
            self.cos_cached = new_cos
            self.sin_cached = new_sin
        else:
            self.cos_cached = torch.cat([self.cos_cached, new_cos], dim=0)
            self.sin_cached = torch.cat([self.sin_cached, new_sin], dim=0)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        """
        前向传播函数


        参数说明：
            x (torch.Tensor): 输入张量，形状为 (..., seq_len, d_k)
            token_positions (torch.Tensor): token位置信息，形状可以是：
                - (seq_len,): 单个序列的位置
                - (batch, seq_len): 批次中每个序列的位置
                - None: 自动生成0到seq_len-1的位置


        返回：
            torch.Tensor: 应用RoPE后的张量，形状与输入相同


        处理步骤：
            1. 验证输入维度
            2. 处理位置信息（转换为long类型）
            3. 确定需要的缓存长度并动态扩展
            4. 从缓存中获取对应的cos/sin值
            5. 应用旋转操作
        """
        assert x.size(-1) == self.d_k, f"Expected last dim d_k={self.d_k}, got {x.size(-1)}"
        seq_len = x.size(-2)

        # 处理位置信息
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            token_positions = token_positions.to(device=x.device, dtype=torch.long)

        # 确定需要的缓存长度
        max_pos = int(token_positions.max().item()) if token_positions.numel() > 0 else 0
        needed_len = max_pos + 1

        # 动态扩展缓存
        self._maybe_extend_cache(needed_len=needed_len, device=x.device)

        # 从缓存中获取对应的cos/sin值
        cos = self.cos_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)
        sin = self.sin_cached.index_select(0, token_positions.reshape(-1)).reshape(*token_positions.shape, -1)

        # 检查序列长度对齐
        if cos.shape[-2] != seq_len:
            raise ValueError(f"token_positions seq dim {cos.shape[-2]} != x seq_len {seq_len}")

        # 数据类型对齐
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        return self._apply_rope(x, cos, sin)


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax函数


    Softmax函数将任意实数向量转换为概率分布，所有元素的和为1。
    常用于分类任务的最后一层，将模型的输出转换为每个类别的概率。


    数学公式：
        softmax(x_i) = exp(x_i) / sum(exp(x_j))


    深度学习小知识：
        - 为什么要减去最大值？
          为了数值稳定性。exp(x)函数增长非常快，如果x的值很大，exp(x)可能会超出浮点数的表示范围（溢出）。
          减去最大值后，exp(x_i - max(x))的最大值变为exp(0)=1，这样可以有效避免溢出。
        - keepdim=True的作用？
          保持维度不变，方便广播运算。


    【PyTorch替换说明】
    可以直接用 torch.nn.functional.softmax 替换：
        import torch.nn.functional as F
        F.softmax(x, dim=dim)


    实现区别：
        - 功能完全等价
    """
    # 为了数值稳定性，减去输入张量在指定维度上的最大值
    x_normalized = x - x.max(dim=dim, keepdim=True).values

    # 计算指数
    x_exp = x_normalized.exp()

    # 归一化：除以指定维度上的所有元素的和
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: torch.Tensor = None) -> torch.Tensor:
    """
    缩放点积注意力（Scaled Dot-Product Attention）


    这是Transformer中注意力机制的核心计算单元。


    数学公式：
        Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) V


    深度学习小知识：
        - Q（Query）、K（Key）、V（Value）是注意力机制的三个输入
        - Q和K进行点积计算相似度
        - 除以sqrt(d_k)进行缩放，防止点积结果过大导致softmax梯度过小
        - softmax将相似度转换为权重
        - 权重与V相乘，得到加权后的Value，即注意力输出
        - mask（掩码）的作用：
          在解码器中，为了防止模型看到未来的信息，需要对未来的位置进行掩码
          将未来位置的注意力分数设置为负无穷，经过softmax后变为0


    【PyTorch替换说明】
    可以直接用 torch.nn.functional.scaled_dot_product_attention 替换（PyTorch 2.0+）：
        import torch.nn.functional as F
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


    实现区别：
        - 功能完全等价，但PyTorch官方实现通常会进行更多优化


    参数说明：
        q (torch.Tensor): 查询向量，形状为 (..., seq_len_q, d_q)
        k (torch.Tensor): 键向量，形状为 (..., seq_len_k, d_k)
        v (torch.Tensor): 值向量，形状为 (..., seq_len_v, d_v)
        mask (torch.Tensor, optional): 注意力掩码，形状为 (seq_len, seq_len) 或 (..., seq_len_q, seq_len_k)
                                       通常是一个布尔张量，True表示允许注意力，False表示禁止注意力


    返回：
        torch.Tensor: 注意力输出，形状为 (..., seq_len_q, d_v)
    """
    # 计算Q和K的点积，并进行缩放
    # q @ k.transpose(-2, -1) 是 QK^T
    # k.size(-1) 是 d_k
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))

    # 应用掩码
    if mask is not None:
        # 将mask中为0的位置（即需要被掩盖的位置）设置为负无穷
        # 经过softmax后，这些位置的权重将变为0
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 应用softmax，得到注意力权重
    attn_weights = softmax(scores, dim=-1)

    # 注意力权重与V相乘，得到最终输出
    return attn_weights @ v


def scaled_dot_product_attention_einsum(q, k, v, mask=None):
    """
    使用einsum实现的缩放点积注意力


    这是scaled_dot_product_attention的另一种实现方式，使用爱因斯坦求和约定（einsum）。


    深度学习小知识：
        - einsum是一种简洁的矩阵运算表示方法
        - '...nd, ...md -> ...nm' 表示：
          - 第一个张量的最后两个维度是n和d
          - 第二个张量的最后两个维度是m和d
          - 输出张量的最后两个维度是n和m
          - 省略号(...)表示前面的维度保持不变
        - 这种写法更简洁，但可读性稍差


    注意：
        这个函数在最终模型中未被使用，仅作为参考实现。


    参数说明：
        q (torch.Tensor): 查询向量，形状为 (..., seq_len_q, d_q)
        k (torch.Tensor): 键向量，形状为 (..., seq_len_k, d_k)
        v (torch.Tensor): 值向量，形状为 (..., seq_len_v, d_v)
        mask (torch.Tensor, optional): 注意力掩码


    返回：
        torch.Tensor: 注意力输出，形状为 (..., seq_len_q, d_v)
    """
    # 使用einsum计算QK^T
    scores = torch.einsum('...nd, ...md -> ...nm', q, k) / math.sqrt(k.size(-1))

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = softmax(scores, dim=-1)

    # 使用einsum计算注意力输出
    return torch.einsum('...nm, ...mv -> ...nv', attn_weights, v)


class CausalSelfAttention(nn.Module):
    """
    因果多头自注意力（Causal Multi-Head Self-Attention）


    这是自注意力机制的一种实现，具有因果性（Causal）和多头（Multi-Head）特性。


    深度学习小知识：
        - 因果性：在生成文本时，每个token只能看到它之前的token，不能看到未来的token
          这通过下三角掩码（lower triangular mask）实现
        - 多头注意力：将注意力机制分成多个"头"，每个头关注不同的信息
          例如：一个头可能关注语法结构，另一个头可能关注语义关系
        - 多头注意力的好处：
          1. 可以并行计算
          2. 每个头可以学习不同的特征
          3. 增强模型的表达能力


    注意：
        这个类在最终模型中未被使用，仅作为参考实现。
        最终模型使用的是 CausalSelfAttention_RoPE，它增加了旋转位置编码。


    参数说明：
        d_model (int): 模型的特征维度
        n_head (int): 注意力头的数量
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head

        # key, query, value projections for all heads, but in a batch
        # 使用一个线性层同时计算Q、K、V，效率更高
        # 输出维度是3*d_model，然后分成三份
        self.qkv_proj = Linear(self.d_model, 3 * self.d_model)
        # output projection
        self.out_proj = Linear(self.d_model, self.d_model)

    def forward(self, x: torch.Tensor):
        """
        前向传播函数


        参数说明：
            x (torch.Tensor): 输入张量，形状为 (batch, seq_len, d_model)


        返回：
            torch.Tensor: 输出张量，形状为 (batch, seq_len, d_model)


        处理步骤：
            1. 计算Q、K、V
            2. 将Q、K、V分成多个头
            3. 计算注意力
            4. 合并多个头的输出
            5. 通过输出投影层
        """
        B, T, C = x.size()

        # 计算 key, query, value
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        q, k, v = qkv.split(self.d_model, dim=-1)  # each is (batch, seq_len, d_model)

        # 将Q、K、V分成多个头
        # view: 重塑张量形状
        # transpose: 交换维度，从 (B, T, n_head, head_size) 变为 (B, n_head, T, head_size)
        # 这样每个头可以独立计算注意力
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (batch, n_head, seq_len, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (batch, n_head, seq_len, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (batch, n_head, seq_len, head_size)

        # 创建因果掩码（下三角矩阵）
        # torch.tril返回下三角部分，对角线及以上为True，以下为False
        attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))

        # 计算注意力
        attn_output = scaled_dot_product_attention(q, k, v, mask=attn_mask)  # (batch, n_head, seq_len, head_size)

        # 合并多个头的输出
        # transpose(1, 2): 从 (B, n_head, T, head_size) 变为 (B, T, n_head, head_size)
        # contiguous(): 确保张量在内存中是连续存储的（某些操作需要）
        # view(B, T, C): 重塑为 (B, T, d_model)，相当于将所有头的输出拼接起来
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (batch, seq_len, d_model)

        # 输出投影
        y = self.out_proj(attn_output)  # (batch, seq_len, d_model)

        return y


class CausalSelfAttention_RoPE(nn.Module):
    """
    带RoPE的因果多头自注意力（Causal Multi-Head Self-Attention with RoPE）


    这是最终模型中使用的注意力机制，它在CausalSelfAttention的基础上，
    将旋转位置编码（RoPE）应用到Query和Key向量上。


    深度学习小知识：
        - 结合了RoPE的优点，可以更好地处理长序列的相对位置信息
        - 避免了传统位置编码在序列长度超过训练长度时性能下降的问题


    参数说明：
        d_model (int): 模型的特征维度
        n_head (int): 注意力头的数量
        theta (float): RoPE的频率参数，默认为10000.0
    """

    def __init__(self, d_model: int, n_head: int, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head  # 每个注意力头的维度

        # key, query, value projections for all heads, but in a batch
        self.qkv_proj = Linear(self.d_model, 3 * self.d_model)
        # output projection
        self.out_proj = Linear(self.d_model, self.d_model)

        # RoPE层，注意d_k设置为head_dim
        self.rope = RoPE(theta=theta, d_k=self.head_dim)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        """
        前向传播函数


        参数说明：
            x (torch.Tensor): 输入张量，形状为 (batch, seq_len, d_model)
            token_positions (torch.Tensor, optional): token位置信息


        返回：
            torch.Tensor: 输出张量，形状为 (batch, seq_len, d_model)


        处理步骤：
            1. 计算Q、K、V
            2. 将Q、K、V分成多个头
            3. 对Q、K应用RoPE
            4. 创建因果掩码
            5. 计算注意力
            6. 合并多个头的输出
            7. 通过输出投影层
        """
        B, T, C = x.size()

        # 计算 key, query, value
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        # 将qkv分成q, k, v
        q, k, v = qkv.split(self.d_model, dim=-1)  # each is (batch, seq_len, d_model)

        # 将Q、K、V分成多个头，并转置维度
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (batch, n_head, seq_len, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (batch, n_head, seq_len, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (batch, n_head, seq_len, head_size)

        # 处理token_positions，确保其形状正确
        if token_positions is None:
            # 如果没有提供位置信息，则生成一个从0到T-1的序列
            token_positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1,T)
        elif token_positions.ndim == 1:
            # 如果是1维，增加一个批次维度
            token_positions = token_positions.unsqueeze(0)  # (1,T)

        # 对Q和K应用旋转位置编码
        q = self.rope(q, token_positions)  # (B,H,T,hd)
        k = self.rope(k, token_positions)  # (B,H,T,hd)

        # 创建因果掩码（下三角矩阵）
        attn_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))

        # 计算注意力
        attn_output = scaled_dot_product_attention(q, k, v, mask=attn_mask)  # (batch, n_head, seq_len, head_size)

        # 合并多个头的输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (batch, seq_len, d_model)

        # 输出投影
        y = self.out_proj(attn_output)  # (batch, seq_len, d_model)

        return y


class Block(nn.Module):
    """
    Transformer Block（Transformer块）


    Transformer的基本构建块，包含两个主要组件：
        1. 多头自注意力机制（带残差连接和层归一化）
        2. 前馈网络（带残差连接和层归一化）


    深度学习小知识：
        - 残差连接（Residual Connection）：
          将输入直接加到输出上，形成 x = x + f(x) 的结构
          好处：
          1. 缓解梯度消失问题
          2. 让网络更容易学习恒等映射
          3. 提高训练稳定性


        - Pre-LN（Pre-Normalization）架构：
          先进行层归一化，再进行注意力/前馈计算
          相比Post-LN（先计算后归一化）有更好的训练稳定性


    参数说明：
        d_model (int): 模型的特征维度
        n_head (int): 注意力头的数量
        d_ff (int): 前馈网络的中间层维度
        theta (float): RoPE的频率参数，默认为10000.0
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, theta: float = 10000.0):
        super().__init__()
        # Pre-LN架构：先归一化，再计算注意力
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention_RoPE(d_model, n_head, theta)

        # Pre-LN架构：先归一化，再计算前馈网络
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播函数


        参数说明：
            x (torch.Tensor): 输入张量，形状为 (batch, seq_len, d_model)
            token_positions (torch.Tensor, optional): token位置信息


        返回：
            torch.Tensor: 输出张量，形状为 (batch, seq_len, d_model)


        处理步骤：
            1. 注意力子层：x = x + Attention(Norm(x))
            2. 前馈子层：x = x + FFN(Norm(x))
        """
        # 注意力子层：残差连接 + Pre-LN
        # 先归一化，再计算注意力，最后加上原始输入
        x = x + self.attn(self.attn_norm(x), token_positions=token_positions)

        # 前馈子层：残差连接 + Pre-LN
        # 先归一化，再计算前馈网络，最后加上注意力子层的输出
        x = x + self.ffn(self.ffn_norm(x))

        return x


class DepthAttentionResidual(nn.Module):
    """
    教学版注意力残差（Attention Residual）模块。

    传统残差连接只做一件事：把“上一层的输入”原样加回来。
    例如：x = x + f(x)。这很像写代码时只参考上一步的结果。

    注意力残差的想法是：当前层不只看上一层，而是可以回头看更早的多层历史。
    例如：embedding输出、第1层输出、第2层输出都可以作为候选残差信息。
    模型会用一组注意力权重决定：当前这个token更应该相信哪一层的历史表示。

    这里实现的是简化版 depth-wise attention：
    - 普通自注意力是在“时间/序列维度”上选择前面哪些token重要。
    - 这里是在“深度/层数维度”上选择前面哪些层重要。
    """

    def __init__(self, d_model: int, window: int = 0):
        super().__init__()
        self.window = window

        # query_proj把当前层输入变成“我要找什么信息”的向量。
        # key_proj把历史层输出变成“我这里有什么信息”的向量。
        # 注意：value这里直接使用历史状态本身，不再投影，这样当只有一个历史状态时，
        # 注意力残差会退化成标准残差，更容易稳定训练和理解。
        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)

    def forward(self, current_x: torch.Tensor, previous_states: list[torch.Tensor]) -> torch.Tensor:
        """
        根据当前层输入，从历史层输出中加权混合出一个残差基底。

        参数说明：
            current_x: 当前层输入，形状是 (batch, seq_len, d_model)
            previous_states: 历史层输出列表，每个元素形状也是 (batch, seq_len, d_model)

        返回：
            mixed_residual: 从历史层中“注意力挑选”出来的残差，形状仍是 (batch, seq_len, d_model)
        """
        if self.window > 0:
            previous_states = previous_states[-self.window:]

        # 假设有3个历史状态，每个都是 (B, T, C)。
        # stack后变成 (B, T, 3, C)，新增的第3维就是“层深度维度”。
        # 对小白来说，可以把它理解成：每个token位置都有一个历史版本列表。
        history = torch.stack(previous_states, dim=2)

        # 当前层生成query，形状 (B, T, C)，表示“当前token想找什么层信息”。
        query = self.query_proj(current_x)

        # 历史层生成key，形状 (B, T, D, C)，D是可回看的历史层数量。
        # key表示“每个历史层能提供什么信息”。
        key = self.key_proj(history)

        # query.unsqueeze(2)把query扩展成 (B, T, 1, C)，方便和每个历史层key相乘。
        # 乘完再sum，相当于计算“当前query”和“每个历史层key”的相似度。
        scores = (query.unsqueeze(2) * key).sum(dim=-1) / math.sqrt(query.size(-1))

        # softmax把相似度变成权重，所有历史层权重加起来等于1。
        # 权重大表示：当前token更应该从这一层历史表示中拿残差信息。
        weights = softmax(scores, dim=-1)

        # 用权重对历史状态加权求和，得到新的残差基底。
        # 如果某一层权重是0.8，它对最终残差的影响就更大。
        mixed_residual = (weights.unsqueeze(-1) * history).sum(dim=2)
        return mixed_residual


class AttentionResidualBlock(nn.Module):
    """
    使用注意力残差的Transformer块。

    它和普通Block一样，仍然包含：
    1. 带RoPE的因果自注意力
    2. SwiGLU前馈网络
    3. RMSNorm预归一化

    不同点只在残差来源：
    - 普通Block：残差就是上一层输入x。
    - AttentionResidualBlock：残差来自多个历史层的注意力加权混合。
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, theta: float = 10000.0,
                 attn_residual_window: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention_RoPE(d_model, n_head, theta)
        self.attn_residual = DepthAttentionResidual(d_model, window=attn_residual_window)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ffn_residual = DepthAttentionResidual(d_model, window=attn_residual_window)

    def forward(self, x: torch.Tensor, previous_states: list[torch.Tensor],
                token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播。

        previous_states保存了embedding输出和前面各层输出。
        当前层会先用注意力从这些历史状态中混合出残差，再加上当前子层的计算结果。
        """
        attn_base = self.attn_residual(x, previous_states)
        x = attn_base + self.attn(self.attn_norm(x), token_positions=token_positions)

        # 注意力子层输出后的x也是有用历史，因此FFN残差可以把它也纳入候选。
        ffn_base = self.ffn_residual(x, previous_states + [x])
        x = ffn_base + self.ffn(self.ffn_norm(x))

        return x


class Transformer(nn.Module):
    """
    Transformer语言模型


    完整的Transformer语言模型，包含：
        1. 词嵌入层
        2. 多层Transformer块
        3. 最终层归一化
        4. 语言模型头部（输出投影层）


    深度学习小知识：
        - 语言模型（Language Model）的任务：
          给定前面的token，预测下一个token的概率分布
        - 生成过程：
          1. 输入token序列
          2. 模型输出下一个token的概率分布
          3. 根据概率分布采样或选择最可能的token
          4. 将新token添加到输入序列中，重复过程


    参数说明：
        d_model (int): 模型的特征维度
        n_head (int): 注意力头的数量
        d_ff (int): 前馈网络的中间层维度
        theta (float): RoPE的频率参数
        vocab_size (int): 词表大小
        context_length (int): 最大上下文长度
        num_layers (int): Transformer块的层数
    """

    def __init__(self, d_model: int, n_head: int, d_ff: int, theta: float, vocab_size: int, context_length: int,
                 num_layers: int, residual_type: str = "standard", attn_residual_window: int = 0):
        super().__init__()
        self.residual_type = residual_type
        self.attn_residual_window = attn_residual_window

        if residual_type == "standard":
            self.layers = nn.ModuleList([
                Block(d_model, n_head, d_ff, theta)
                for _ in range(num_layers)
            ])
        elif residual_type == "attention":
            self.layers = nn.ModuleList([
                AttentionResidualBlock(d_model, n_head, d_ff, theta, attn_residual_window)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError("residual_type must be 'standard' or 'attention'")

        # 最终层归一化
        self.norm = RMSNorm(d_model)

        # 最大上下文长度
        self.context_length = context_length

        # 词嵌入层
        self.embedding = Embedding(vocab_size, d_model)

        # 语言模型头部（输出投影层）
        # 将d_model维的向量映射回vocab_size维，表示每个token的概率
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        前向传播函数


        参数说明：
            x (torch.Tensor): 输入token ID张量，形状为 (batch, seq_len)
            token_positions (torch.Tensor, optional): token位置信息


        返回：
            torch.Tensor: 输出logits，形状为 (batch, seq_len, vocab_size)
                        表示每个位置每个token的未归一化概率（logits）


        处理步骤：
            1. 检查序列长度是否超过最大上下文长度
            2. 将token ID转换为嵌入向量
            3. 通过所有Transformer块
            4. 最终归一化
            5. 通过语言模型头部得到logits
        """
        B, T = x.shape

        # 检查序列长度是否超过最大上下文长度
        assert T <= self.context_length, f"Cannot forward sequence of length {T}, context length is only {self.context_length}"

        # 将token ID转换为嵌入向量
        x = self.embedding(x)

        if self.residual_type == "standard":
            for layer in self.layers:
                x = layer(x, token_positions=token_positions)
        else:
            residual_states = [x]
            for layer in self.layers:
                x = layer(x, residual_states, token_positions=token_positions)
                residual_states.append(x)

        # 最终归一化
        x = self.norm(x)

        # 通过语言模型头部得到logits
        logits = self.lm_head(x)

        return logits
