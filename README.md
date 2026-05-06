# llm-from-scratch

这是一个从零实现的 Decoder-only Transformer 语言模型与 BPE 分词器项目。项目目标不是调用高层封装快速搭建一个模型，而是尽可能把现代大语言模型中的关键组件拆开实现，让训练流程、模型结构、分词器、优化器、学习率调度、checkpoint 等环节都可以被清晰地观察和修改。

本项目可以训练，不只是静态代码示例。当前默认配置面向 TinyStories 这种小型叙事数据集，目标是训练一个能够生成流利英文故事或简单对话的小型语言模型。

## 项目核心理念

本项目刻意避免依赖高层 `torch.nn` 模块，重点展示 Transformer 内部机制。

项目中避免直接使用的高层组件包括：

- `nn.Linear`
- `nn.LayerNorm`
- `nn.Transformer`
- `nn.MultiheadAttention`
- 其他会隐藏核心计算细节的高层封装

允许使用的 PyTorch 能力主要是：

- `torch.Tensor` 与基础张量计算
- `torch.nn.Parameter`，用于注册可训练参数
- `torch.nn.Module` / `torch.nn.ModuleList`，用于组织模型结构
- `torch.optim.Optimizer`，仅作为自定义 AdamW 的基类

这样做的好处是：

- 可以真正理解 Embedding、Attention、FFN、Norm、Residual 的数据流。
- 可以明确看到每个参数矩阵如何初始化、如何参与前向传播。
- 可以更容易定位训练中的 loss 异常、梯度异常、token id 越界等问题。
- 可以作为学习现代 LLM 架构的可运行实验平台。

## 当前使用的数据集

当前项目默认围绕 TinyStories 数据集进行实验。

典型文件命名为：

```text
tokenized_data/TinyStories-train.txt
tokenized_data/TinyStories-valid.txt
```

这些原始 `.txt` 文件不能直接送进模型训练，需要先完成两步：

1. 使用训练集文本训练 BPE 分词器，得到 vocab 和 merges。
2. 使用训练好的 BPE 分词器把 train/valid 文本编码成 `.bin` token id 文件。

模型训练脚本默认读取的是：

```text
tokenized_data/your_train_data.bin
tokenized_data/your_val_data.bin
```

也就是说，`.txt` 是原始语料，`.bin` 才是模型训练真正使用的数据。

## 项目结构

```text
├── main/
│   ├── model.py                  # Transformer、Attention、RoPE、RMSNorm、SwiGLU、注意力残差等核心模型组件
│   ├── tokenizer_optimized.py     # 加载和使用自定义 BPE 分词器
│   ├── train_model.py            # AdamW、学习率调度、梯度裁剪、batch、checkpoint 等训练工具
│   ├── run_train_model.py        # 模型训练主入口
│   ├── prepare_tokenized_data.py # txt -> bin 数据预处理脚本
│   └── play_model.ipynb          # 加载 checkpoint 进行文本生成
├── bpe-optimized-from-scratch/
│   └── bpe_optimized/
│       └── train_bpe_parallel_optimized.py # 优化版 BPE 训练脚本
├── tokenized_data/               # 原始文本与编码后的 .bin 数据
├── trained_tokenizer/            # 训练好的 tokenizer vocab 与 merges
├── train_logs/                   # 每次训练的日志、metrics.csv、checkpoint
├── img/                          # 架构图
├── run.sh                        # Linux/macOS Shell 训练启动脚本
├── run.py                        # 与 run.sh 参数一致的 Python 训练启动脚本
├── requirements.txt              # 依赖列表
└── FAQ.md                        # 架构问题与设计取舍说明
```

## 模型架构概览

当前模型是 Decoder-only Transformer，整体流程如下：

```text
token ids
  -> Token Embedding
  -> 多层 Transformer Block
      -> RMSNorm
      -> Causal Self-Attention + RoPE
      -> Residual Connection
      -> RMSNorm
      -> SwiGLU FFN
      -> Residual Connection
  -> Final RMSNorm
  -> LM Head
  -> logits
```

架构图：

```text
img/architecture.png
```

如果你的 Markdown 查看器支持图片，可以打开该文件查看更直观的结构图。

## 已实现的核心模型组件

### 1. 自定义 Linear

项目实现了自己的线性层，而不是直接使用 `nn.Linear`。

这能帮助理解语言模型里最基础的矩阵乘法：

```text
y = xW^T
```

同时也能明确控制权重初始化、bias 使用策略和参数注册方式。

### 2. Token Embedding

Embedding 层负责把离散 token id 映射为连续向量表示。

训练前必须保证：

```text
0 <= token_id < vocab_size
```

如果 `.bin` 数据中出现 `token_id >= vocab_size`，CPU 上常见错误是 `index out of range`，CUDA 上则可能表现为 `device-side assert triggered`。当前训练脚本已经加入 token id 范围校验，会在进入 Embedding 前提前报出清晰错误。

### 3. Pre-Norm + RMSNorm

本项目采用 Pre-Norm 结构：

```text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

相比 Post-Norm，Pre-Norm 在深层网络中通常更稳定，梯度更容易传递。

RMSNorm 相比 LayerNorm 更简洁，只根据均方根进行缩放，计算成本更低，也符合许多现代 LLM 的设计方向。

### 4. RoPE 旋转位置编码

项目实现了 RoPE，用来给 Attention 注入位置信息。

RoPE 的核心思想不是把位置向量直接加到 token embedding 上，而是在 Query 和 Key 上做旋转变换，让 Attention 分数天然包含相对位置信息。

当前训练配置中常用：

```text
--theta 10000
--context_length 256
```

### 5. Causal Self-Attention

模型使用因果自注意力，保证当前位置只能看到当前位置及之前的 token，不能偷看未来 token。

这正是自回归语言模型可以逐 token 生成文本的基础。

### 6. SwiGLU 前馈网络

FFN 部分采用 SwiGLU，而不是传统 ReLU MLP。

SwiGLU 是现代大模型中常见的门控前馈结构，表达能力通常强于普通激活函数，适合语言建模任务。

### 7. 标准残差连接

默认残差连接是标准 Transformer 形式：

```text
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

它的作用是给梯度提供更短路径，缓解深层网络训练中的梯度消失问题，也让每个子层只需要学习对当前表示的“增量修正”。

## 新增：注意力残差连接

项目已经新增一个可选的注意力残差模块，可以在训练时通过参数切换。

默认仍然使用标准残差：

```bash
--residual_type standard
--attn_residual_window 0
```

如果想尝试注意力残差：

```bash
--residual_type attention
--attn_residual_window 0
```

### 注意力残差做了什么

标准残差只把上一层输入原样加回来：

```text
residual = x
x = residual + 子层输出
```

注意力残差则会从多个历史层状态中，用 attention 计算一个更灵活的残差基底：

```text
previous_states = [embedding输出, 第1层输出, 第2层输出, ...]
residual = AttentionMix(previous_states)
x = residual + 子层输出
```

本项目实现的是教学友好的 depth-wise attention residual：

- 不是沿 token 时间维度做 attention。
- 而是沿“层深度”维度做 attention。
- 对每个 token 位置，模型会学习当前层更应该参考哪一个历史层输出。

### `attn_residual_window` 的含义

`attn_residual_window` 控制注意力残差最多回看多少个历史层状态。

```text
--attn_residual_window 0
```

表示使用全部历史层状态。

```text
--attn_residual_window 4
```

表示最多只使用最近 4 个历史层状态。

窗口越大，信息来源越多，但显存和计算开销也越高。

## 为什么本项目默认不推荐使用注意力残差

虽然注意力残差很适合作为架构实验，但当前项目默认不推荐把它作为主训练路径，原因如下：

1. TinyStories 任务较小

   TinyStories 主要考察小模型能否学会基本语法、叙事结构和简单常识。对于这个目标，标准残差已经足够稳定。

2. 当前默认模型层数不深

   默认配置大约是 4 层 Transformer。注意力残差更可能在很深的模型中体现价值，在浅层小模型里收益通常不明显。

3. 会增加参数量和计算量

   注意力残差需要额外的 Query/Key 投影和历史状态混合，训练单步会变慢，显存占用也会上升。

4. 会让 checkpoint 不通用

   `residual_type=standard` 和 `residual_type=attention` 的参数结构不同。用标准残差训练出的 checkpoint 不能直接按注意力残差结构加载，反过来也一样。

5. 本项目定位优先是从零理解 Transformer

   标准残差是理解 Transformer 的基础。注意力残差适合作为进阶实验，而不是初学主线。

因此推荐顺序是：

```text
先使用 standard 跑通完整训练流程
再使用 attention 做对照实验
```

## 与 Qwen3 架构的关系

本项目不是 Qwen3 的完整复刻，而是参考了 Qwen3 和现代 Decoder-only LLM 中适合小模型教学训练的设计。

参考 FAQ.md 中的整理，Qwen3 相比传统 Decoder-only 模型的特点主要包括：

- Pre-Norm，提高深层训练稳定性。
- RMSNorm，替代 LayerNorm，减少计算量。
- RoPE，增强位置信息建模能力。
- SwiGLU，提升 FFN 表达能力。
- QK-Norm，在超大规模训练中稳定 Attention 分数。
- GQA，减少 KV Cache 显存占用。
- MoE，在大模型中用稀疏专家提高参数容量和推理效率。
- 长上下文训练策略，支持更长上下文窗口。

当前项目选择实现并默认启用的是：

```text
Pre-Norm + RMSNorm + RoPE + SwiGLU + Dense FFN
```

当前项目没有默认引入完整 Qwen3 MoE/GQA/QK-Norm 体系，原因是：

- TinyStories 不需要复杂 MoE 路由。
- 小模型更需要稳定、简单、可解释的 dense 结构。
- GQA 的主要收益在大 batch、大上下文推理时更明显。
- QK-Norm 更适合后续作为单独实验加入，而不是和所有优化一次性耦合。

换句话说，本项目采用的是“Qwen3-Mini-Dense 风格”的取舍：保留适合小模型预训练的稳定组件，暂时舍弃大规模集群训练才更需要的复杂组件。

## 为什么不用 DeepSeek V4 架构

DeepSeek V4 类型的架构更偏向大规模、高复杂度、高工程优化路线，常见特点包括：

- 更复杂的 MoE 设计。
- 更激进的稀疏激活比例。
- 更复杂的注意力压缩或混合注意力机制。
- 更高的训练工程和调参难度。
- 对大规模数据、大规模参数和分布式训练更友好。

但本项目当前目标是：

```text
在 TinyStories 上从零预训练一个能生成流利文本的小型 Transformer
```

在这个目标下，DeepSeek V4 风格并不合适：

1. 架构过重

   小数据集很难把复杂 MoE 路由训练充分，容易出现专家使用不均衡或专家塌陷。

2. 教学成本过高

   如果一开始就引入复杂 MoE、压缩注意力和更激进的训练技巧，反而会掩盖 Transformer 最核心的机制。

3. 收益不匹配

   TinyStories 的核心需求是语言流利度、基础语法和叙事连贯性，不需要超大模型容量和复杂稀疏路由。

4. Debug 难度更高

   从零实现项目更容易遇到 shape、mask、初始化、token id、梯度等底层问题。Dense Transformer 更容易定位问题。

因此，本项目选择 Qwen3 风格的稳定小模型路线，而不是 DeepSeek V4 风格的重型 MoE 路线。

## BPE 分词器与优化创新点

项目包含一个从零实现并优化过的 BPE tokenizer 训练脚本。

核心脚本：

```bash
python bpe-optimized-from-scratch/bpe_optimized/train_bpe_parallel_optimized.py \
  --input_path tokenized_data/TinyStories-train.txt \
  --vocab_size 10000
```

默认输出：

```text
trained_tokenizer/vocab_of_your_tokenizer.json
trained_tokenizer/merges_of_your_tokenizer.json
```

BPE 训练侧的主要优化包括：

### 1. Weighted Space Reduction

把重复出现的 pre-token 压缩成带权重的统计单元，避免对相同字符串重复计算 pair 频率。

### 2. Inverted Indexing

维护 pair 到相关 token 的反向索引。每次 merge 后只更新受影响的局部区域，而不是全量扫描所有 token。

### 3. Lazy Max-Heap

使用最大堆快速找到当前最高频 pair，同时用 lazy validation 避免每次频率变化都重建整个堆。

### 4. Memoization

缓存重复计算结果，减少正则切分、bytes 转换、pair 统计中的重复开销。

### 5. Rank Dictionary

编码阶段使用 merge rank 字典快速判断 pair 合并优先级。

### 6. Vectorized Mapping

在部分映射和转换步骤中使用更高效的数据结构与批处理方式，减少 Python 层循环开销。

### 7. 进度条

BPE 脚本已经加入预分词和 merge 学习阶段的进度条，运行时可以看到当前处理进度。

### 8. vocab 越界保护

BPE 保存前会校验 vocab id 是否连续且不超过指定 `vocab_size`，避免生成 `id == vocab_size` 这种会导致 Embedding 越界的 tokenizer。

## 数据预处理流程

训练模型前，需要先把 `.txt` 编码成 `.bin`。

```bash
python main/prepare_tokenized_data.py \
  --train_txt tokenized_data/TinyStories-train.txt \
  --val_txt tokenized_data/TinyStories-valid.txt
```

默认输出：

```text
tokenized_data/your_train_data.bin
tokenized_data/your_val_data.bin
```

当前 `.bin` 使用 `np.uint16` 保存 token id，因此推荐 `vocab_size <= 65536`。默认 `vocab_size=10000` 是安全的。

如果重新训练 tokenizer，必须重新生成 `.bin` 文件。否则 tokenizer vocab 与 `.bin` token id 可能不匹配，训练时会报 token id 越界。

## 模型训练

### 使用 Shell 脚本

```bash
bash run.sh
```

### 使用 Python 脚本

```bash
python run.py
```

`run.py` 与 `run.sh` 的功能和参数保持一致，适合不方便使用 shell 的环境。

默认训练参数包括：

```text
batch_size=64
max_iters=4200
eval_interval=100
eval_iters=20
vocab_size=10000
context_length=256
n_head=16
n_layers=4
d_model=512
d_ff=1344
residual_type=standard
attn_residual_window=0
max_lr=6e-4
min_lr=6e-5
warmup_iters=200
lr_decay_iters=3600
```

## 如何查看训练情况

每次训练会在 `train_logs/` 下创建一个新的运行目录：

```text
train_logs/run_YYYYMMDD_HHMMSS/
```

其中常见文件包括：

```text
train.log
metrics.csv
ckpt_iter_*.pt
```

查看日志：

```bash
tail -f train_logs/run_具体时间/train.log
```

重点观察：

- `loss`：当前训练 batch 的损失。
- `train loss`：训练集估计损失。
- `val loss`：验证集估计损失。
- `lr`：当前学习率。
- `grad_norm`：梯度范数。
- `[Generated at iter ...]`：训练中定期生成的文本样例。

判断训练是否正常，可以看：

- loss 是否整体下降。
- train loss 和 val loss 是否差距过大。
- grad_norm 是否频繁爆炸。
- 生成文本是否从乱码逐渐变成更连贯的词和句子。

## checkpoint 使用方式

训练脚本会定期保存 checkpoint：

```text
train_logs/run_具体时间/ckpt_iter_XXXX.pt
```

checkpoint 中包含：

- model state
- optimizer state
- 当前 iteration

### 从 checkpoint 继续训练

```bash
python main/run_train_model.py \
  --train_data tokenized_data/your_train_data.bin \
  --val_data tokenized_data/your_val_data.bin \
  --tokenizer_vocab trained_tokenizer/vocab_of_your_tokenizer.json \
  --tokenizer_merges trained_tokenizer/merges_of_your_tokenizer.json \
  --vocab_size 10000 \
  --resume train_logs/run_具体时间/ckpt_iter_XXXX.pt
```

实际使用时可以直接在 `run.sh` 或 `run.py` 中加入 `--resume` 参数。

### 用 checkpoint 生成文本

使用：

```text
main/play_model.ipynb
```

注意：notebook 中的模型参数必须和 checkpoint 训练时一致，尤其是：

```text
vocab_size
context_length
n_head
n_layers
d_model
d_ff
residual_type
attn_residual_window
```

如果 checkpoint 是用：

```text
--residual_type attention
```

训练的，那么 notebook 里也必须设置：

```python
residual_type="attention"
```

否则 `state_dict` 参数结构对不上，无法正确加载。

## 训练稳定性设计

项目中已经包含多项稳定训练设计：

- Pre-Norm，减少深层训练梯度不稳定。
- RMSNorm，降低归一化计算复杂度。
- 梯度裁剪，缓解梯度爆炸。
- Cosine learning rate schedule，让学习率平滑衰减。
- Warmup，避免训练初期学习率过大。
- AdamW，使用解耦权重衰减。
- token id 范围校验，提前发现 tokenizer/bin/vocab_size 不匹配。
- checkpoint 保存与恢复，支持中断后续训。

## 本项目的优化创新点总结

### 模型侧

- 从零实现 Decoder-only Transformer 核心模块。
- 使用 Pre-Norm + RMSNorm 提升稳定性。
- 使用 RoPE 替代传统绝对位置编码。
- 使用 SwiGLU 提升 FFN 表达能力。
- 自定义 AdamW、学习率调度和梯度裁剪。
- 新增可选注意力残差连接，支持标准残差与实验残差切换。
- 训练脚本加入 token id 越界校验，减少 CUDA 难定位错误。

### 分词器侧

- 从零实现 BPE tokenizer。
- 支持并行预分词。
- 使用反向索引减少 merge 更新成本。
- 使用 lazy heap 加速最高频 pair 选择。
- 使用缓存减少重复计算。
- 输出格式与项目 tokenizer 加载逻辑兼容。
- 修复并防护 vocab off-by-one 越界问题。

### 工程侧

- 提供 `run.sh` 和 `run.py` 两种训练入口。
- 每次训练自动创建独立日志目录。
- 保存 `metrics.csv` 便于后续画图分析。
- 定期生成文本样例，方便观察语言能力变化。
- 支持 checkpoint 保存、加载和续训。
- 提供 notebook 加载 checkpoint 进行交互式生成。

## 推荐实验顺序

建议按下面顺序运行整个项目：

### 1. 训练 BPE tokenizer

```bash
python bpe-optimized-from-scratch/bpe_optimized/train_bpe_parallel_optimized.py \
  --input_path tokenized_data/TinyStories-train.txt \
  --vocab_size 10000
```

### 2. 生成训练用 bin 文件

```bash
python main/prepare_tokenized_data.py \
  --train_txt tokenized_data/TinyStories-train.txt \
  --val_txt tokenized_data/TinyStories-valid.txt
```

### 3. 开始模型训练

```bash
bash run.sh
```

或：

```bash
python run.py
```

### 4. 查看训练日志

```bash
tail -f train_logs/run_具体时间/train.log
```

### 5. 使用 checkpoint 生成文本

打开：

```text
main/play_model.ipynb
```

并把 checkpoint 路径改成你实际训练出的 `ckpt_iter_*.pt`。

## 常见问题

### 1. 为什么 GPU 利用率不高

BPE tokenizer 训练主要是 CPU 任务，涉及文件读取、正则切分、字典统计、堆更新和 bytes 操作，GPU 通常不会参与。

模型训练阶段才会主要使用 GPU。

### 2. 为什么会出现 token id out of range

通常原因是：

- tokenizer 训练时的 `vocab_size` 和模型训练时的 `--vocab_size` 不一致。
- 重新训练了 tokenizer，但没有重新生成 `.bin`。
- BPE vocab 曾经出现 off-by-one，生成了 `id == vocab_size` 的非法 token。

当前 BPE 脚本和训练脚本都已经加入对应防护。

### 3. 为什么 loss 下降但生成文本还不好

语言模型通常先学到局部 token 统计，再逐步学会词、短语、句子和长程结构。训练早期 loss 下降不代表生成马上流利，需要结合验证集 loss 和定期生成样例一起判断。

### 4. 是否应该一开始就使用注意力残差

不建议。建议先用标准残差跑通完整流程，再把注意力残差作为对照实验。

### 5. 是否应该把模型改成 MoE

当前不建议。TinyStories 小模型目标更适合 Dense Transformer。MoE 会显著增加实现复杂度和训练不稳定因素，不利于当前项目的学习和调试目标。

## 致谢

- Stanford CS336：本项目的从零实现理念和训练流程参考了该课程的方向。
- Xuying Li：感谢对 CS336 课程的推荐。

## License

MIT
