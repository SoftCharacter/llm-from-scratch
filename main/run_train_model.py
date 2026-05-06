"""
主训练脚本：在用户提供的数据上训练基于Transformer的语言模型


深度学习小知识：
   - 语言模型训练流程：
     1. 准备数据
     2. 初始化模型和优化器
     3. 训练循环
     4. 评估和保存检查点
   - 关键组件：
     - 分词器（Tokenizer）：将文本转换为token ID
     - 模型（Transformer）：核心神经网络架构
     - 优化器（AdamW）：参数更新算法
     - 学习率调度器：动态调整学习率


使用说明：
   可以通过命令行参数配置训练过程，例如：
   python run_train_model.py --train_data train.bin --val_data val.bin --vocab_size 10000


【替代方案说明】
   如果不使用自定义BPE分词器，可以使用tiktoken：
       import tiktoken
       tokenizer = tiktoken.get_encoding("gpt2")
   注意：
       1. 确保分词器词表大小与模型词表大小匹配（GPT2为50257）
       2. 如果使用tiktoken，需要修改generate函数
"""


import os
import time
import math
import argparse
import torch
import numpy as np
import csv
import wandb


from tokenizer_optimized import Tokenizer
# -> If you do not want to use my BPE tokenizer, you can use tiktoken instead
# import tiktoken
# tokenizer = tiktoken.get_encoding("gpt2")
# Note 1: Make sure the tokenizer vocab size matches the model's vocab size (50257 for GPT2) - set it in the `run.sh`
# Note 2: If you use tiktoken, you need to modify the `generate` function accordingly


from train_model import (
   cross_entropy, AdamW, lr_cosine_schedule,
   gradient_clipping, get_batch, save_checkpoint, load_checkpoint
)


from model import Transformer as Model
from model import softmax


tokenizer = None


# 自动检测可用设备
# 深度学习小知识：
#   - CUDA：NVIDIA GPU计算平台
#   - MPS：Apple Silicon GPU计算平台
#   - CPU：中央处理器（备用）
device = "cpu"
if torch.cuda.is_available():
   device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # macOS M-series GPU
   device = "mps"
print(f"using device: {device}")


def parse_args():
   parser = argparse.ArgumentParser(description="Train a model on user-provided data")

   # data and output paths
   parser.add_argument('--train_data', type=str, required=True, help='Path to train.bin (np.memmap)')
   parser.add_argument('--val_data', type=str, required=True, help='Path to val.bin (np.memmap)')
   parser.add_argument('--tokenizer_vocab', type=str, required=True, help='Path to tokenizer vocab file (json)')
   parser.add_argument('--tokenizer_merges', type=str, required=True, help='Path to tokenizer merges file (txt)')
   parser.add_argument('--out_dir', type=str, default='out', help='Directory to save checkpoints')

   # training hyperparameters
   parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
   parser.add_argument('--max_iters', type=int, default=5000, help='Total number of training iterations')
   parser.add_argument('--eval_interval', type=int, default=500, help='Evaluate the model every eval_interval steps')
   parser.add_argument('--eval_iters', type=int, default=200, help='Number of iters in ONE evaluation run')
   parser.add_argument('--log_interval', type=int, default=10, help='Every log_interval steps, log the training loss')


   # model hyperparameters
   parser.add_argument('--vocab_size', type=int, required=True, help='Size of models vocabulary, must align with tokenizer vocab size')
   parser.add_argument('--context_length', type=int, default=256, help='Context length for the model')
   parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
   parser.add_argument('--theta', type=float, default=10000, help='Theta parameter for RoPE')
   parser.add_argument('--n_layers', type=int, default=6, help='Number of transformer layers')
   parser.add_argument('--d_model', type=int, default=512, help='Dimensionality of the model wrt embd space')
   parser.add_argument('--d_ff', type=int, default=1344, help='Dimensionality of the feedforward layer')
   parser.add_argument('--residual_type', choices=['standard', 'attention'], default='standard', help='Residual connection type')
   parser.add_argument('--attn_residual_window', type=int, default=0, help='Number of previous layer states used by attention residual; 0 means all')

   # Optimizer hyperparameters
   parser.add_argument('--weight_decay', type=float, default=1e-1)
   parser.add_argument('--max_norm', type=float, default=1.0, help='Gradient clipping norm')

   # Learning rate schedule parameters
   parser.add_argument('--max_lr', type=float, default=6e-4, help='Maximum learning rate')
   parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')
   parser.add_argument('--warmup_iters', type=int, default=500, help='Number of warm-up iterations')
   parser.add_argument('--lr_decay_iters', type=int, default=5000, help='Number of iterations for learning rate decay')

   # 日志记录
   parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
   parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')


   return parser.parse_args()


def init_tokenizer(vocab_file, merge_file, special_tokens=["<|endoftext|>"]):
   """
   初始化分词器


   深度学习小知识：
       - 分词器需要从文件加载词表和合并规则
       - 特殊token如<|endoftext|>用于标记文本结束


   参数说明：
       vocab_file (str): 词表文件路径
       merge_file (str): 合并规则文件路径
       special_tokens (List[str]): 特殊token列表
   """
   global tokenizer
   tokenizer = Tokenizer.from_files(vocab_file, merge_file, special_tokens)


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
   """
   评估模型在数据集上的平均损失


   深度学习小知识：
       - @torch.no_grad()：禁用梯度计算，节省内存
       - model.eval()：将模型设置为评估模式（影响Dropout、BatchNorm等）
       - model.train()：恢复训练模式


   参数说明：
       model (nn.Module): 要评估的模型
       data: 数据集（训练集或验证集）
       batch_size (int): 批次大小
       context_length (int): 上下文长度
       device: 计算设备
       eval_iters (int): 评估迭代次数


   返回：
       float: 平均损失值
   """
   model.eval()
   losses = torch.zeros(eval_iters)
   for k in range(eval_iters):
       X, Y = get_batch(data, batch_size, context_length, device)  # (B, T)
       logits = model(X)  # logits size (B, T, V)
       loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))  # 等同于 (B*T, V) 以及 (B*T, )
       losses[k] = loss.item()
   model.train()
   return losses.mean()


def validate_token_data(name, data, vocab_size, context_length):
   """
   校验token数据是否能安全送入Embedding层。

   CUDA里的Embedding查表要求所有token ID都在 [0, vocab_size) 范围内。
   如果某个ID大于等于vocab_size，CPU上通常会报index out of range，
   CUDA上则经常表现为device-side assert triggered。
   """
   if len(data) <= context_length:
       raise ValueError(f"{name} is too short: len={len(data)}, context_length={context_length}")

   max_token_id = int(data.max())
   min_token_id = int(data.min())
   if min_token_id < 0 or max_token_id >= vocab_size:
       raise ValueError(
           f"{name} token id out of range: min={min_token_id}, max={max_token_id}, "
           f"but vocab_size={vocab_size}. Please make sure --vocab_size matches tokenizer vocab size "
           f"and regenerate .bin files with the same tokenizer."
       )


@torch.no_grad()
def generate(model, tokenizer, context, max_new_tokens, temperature=1.0, top_p=0.9, eos_id=None, context_length=256, device=None):
   """
   文本生成函数


   深度学习小知识：
       - 文本生成过程：
         1. 给定上下文，模型预测下一个token的概率分布
         2. 从概率分布中采样一个token
         3. 将新token添加到上下文中
         4. 重复步骤1-3，直到达到最大长度或遇到结束符


       - 温度缩放（Temperature Scaling）：
         控制生成的随机性
         - temperature < 1.0：更确定，更保守
         - temperature = 1.0：默认行为
         - temperature > 1.0：更随机，更有创造性


       - Top-p采样（Nucleus Sampling）：
         从累积概率超过p的最小token集合中采样
         例如：p=0.9，选择累积概率达到0.9的最小token集合
         好处：既保证了多样性，又避免了低概率token


   参数说明：
       model (nn.Module): 训练好的模型
       tokenizer (Tokenizer): 分词器
       context (str): 输入上下文文本
       max_new_tokens (int): 最大生成token数
       temperature (float): 温度参数，默认1.0
       top_p (float): Top-p采样阈值，默认0.9
       eos_id (int, optional): 结束符token ID
       context_length (int): 模型支持的最大上下文长度
       device: 计算设备


   返回：
       tuple: (完整句子, 新生成的token)
   """
   model.eval()


   # 将上下文文本编码为token ID
   idx = torch.tensor(tokenizer.encode(context), dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
   generated_tokens = []


   for _ in range(max_new_tokens):
       # 如果当前序列超过模型的上下文长度，截断开头
       idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]


       # 模型前向传播，获取logits
       logits = model(idx_cond)
       # 只取最后一个位置的logits（预测下一个token）
       logits = logits[:, -1, :]  # (B, V)


       # 温度缩放：控制生成的随机性
       logits = logits / max(temperature, 1e-5)


       # Top-p（Nucleus）采样
       if top_p < 1.0:
           # 按概率降序排序
           sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
           # 计算累积概率
           cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1) #对排序后的 logits 应用 softmax，得到概率


           # 找到累积概率超过top_p的token索引（这些token需要被移除）
           # 通过右移掩码，保留累积概率刚好超过top_p的token
           # 强制保留概率最高的token，避免第一个token概率就超过p时移除所有token
           sorted_indices_to_remove = cumulative_probs > top_p
           sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
           sorted_indices_to_remove[..., 0] = 0


           # 将要移除的token的logits设置为负无穷
           for b in range(logits.size(0)):
               indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
               logits[b, indices_to_remove] = -float('Inf')


       # 将logits转换为概率分布 对所有 logits（包括被设置为负无穷的）应用 softmax
       probs = softmax(logits, dim=-1)


       '''
       为什么设计两次归一化？
       为了保证经过top p筛选后的token概率加起来也是1
       1. **先 softmax 得到概率**：计算每个 token 的概率
       2. **计算累积概率**：找到累积概率达到 top_p 的最小 token 集合
       3. **将不在集合中的 token logits 设为 -∞**：这样最后的 softmax 会将这些 token 的概率变为 0
       4. **最后再 softmax**：确保剩余 token 的概率和为 1
      
       ### 如果只用一次 softmax 会怎样？
       probs = softmax(logits, dim=-1)  # 只计算一次
       # ... 筛选后直接把某些概率设为 0
       probs[indices_to_remove] = 0
       # 此时 probs 总和 < 1，需要手动归一化
       probs = probs / probs.sum()  # 必须手动归一化！
       '''


       # 从概率分布中采样一个token
       idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
       generated_tokens.append(idx_next.item())


       # 将新token添加到序列中
       idx = torch.cat((idx, idx_next), dim=1)


       # 如果遇到结束符，停止生成
       if eos_id is not None and (idx_next.item() == eos_id):
           break


   # 返回完整句子和新生成的token
   return tokenizer.decode(idx[0].tolist()), tokenizer.decode(generated_tokens)


def main():
   """
   主函数：模型的训练循环和日志记录
   """
   args = parse_args()
   os.makedirs(args.out_dir, exist_ok=True)


   # 记录训练配置
   print("="*20 + " Training Configurations " + "="*20)
   for arg in vars(args):
       print(f"{arg:20}: {getattr(args, arg)}")
   print("="*65)


   # 初始化指标记录文件
   metrics_path = os.path.join(args.out_dir, "metrics.csv")
   with open(metrics_path, 'w', newline='') as f:
       writer = csv.writer(f)
       writer.writerow(["iter", "train_loss", "val_loss", "lr"])


   # 初始化分词器
   global tokenizer
   if tokenizer is None:
       init_tokenizer(args.tokenizer_vocab, args.tokenizer_merges)


   # 加载数据
   # np.memmap用于内存高效地加载大型数据文件
   train_data = np.memmap(args.train_data, dtype=np.uint16, mode='r')
   val_data = np.memmap(args.val_data, dtype=np.uint16, mode='r')
   validate_token_data("train_data", train_data, args.vocab_size, args.context_length)
   validate_token_data("val_data", val_data, args.vocab_size, args.context_length)


   # 初始化模型和优化器
   model = Model(d_model=args.d_model, n_head=args.n_head, d_ff=args.d_ff, theta=args.theta, vocab_size=args.vocab_size,
                 context_length=args.context_length, num_layers=args.n_layers, residual_type=args.residual_type, attn_residual_window=args.attn_residual_window).to(device)
   # 优化器的初始学习率只是占位符，实际学习率由调度器控制
   optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)


   # 检查点恢复
   start_iter = 0
   if args.resume:
       start_iter = load_checkpoint(args.resume, model, optimizer)
       print(f"Resuming from iteration {start_iter}")


   # 初始化Weights & Biases（用于实验跟踪和可视化）
   if args.use_wandb:
       wandb.init(project="training-260114-orig", config=args)


   # 训练循环
   X, Y = get_batch(train_data, args.batch_size, args.context_length, device)  # 获取初始批次数据
   t0 = time.time()


   for it in range(start_iter, args.max_iters):

       # 更新学习率（余弦退火调度）
       lr = lr_cosine_schedule(it, args.max_lr, args.min_lr, args.warmup_iters, args.lr_decay_iters)
       for param_group in optimizer.param_groups:
           param_group['lr'] = lr


       # 每隔eval_interval步或在最后一步进行评估和日志记录
       last_step = (it == args.max_iters - 1)
       if (it % args.eval_interval == 0) or last_step:
           train_loss = estimate_loss(model, train_data, args.batch_size, args.context_length, device, args.eval_iters)
           val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
           print(f"Iter {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, lr {lr:.2e}")

           # 记录到Weights & Biases
           if args.use_wandb:
               wandb.log({
                   "iter": it,
                   "train/loss": train_loss,
                   "val/loss": val_loss,
                   "lr": lr,
               })

           # 记录到CSV文件
           with open(metrics_path, 'a', newline='') as f:
               writer = csv.writer(f)
               writer.writerow([it, train_loss.item() if torch.is_tensor(train_loss) else train_loss,
                        val_loss.item() if torch.is_tensor(val_loss) else val_loss,
                        lr])

       # 每隔eval_interval * 10步或在最后一步进行文本生成和模型保存
       if (it % (args.eval_interval * 10) == 0 and it > 0) or last_step:
           # 从模型生成文本
           context, temperature, top_p = "Hello, I'm a language model, ", 1.0, 0.9
           full_sentence, new_tokens = generate(
               model,
               tokenizer=tokenizer,
               context=context,
               max_new_tokens=100,
               temperature=temperature,
               top_p=top_p,
               eos_id=tokenizer.special_token_to_id.get("<|endoftext|>"),
               context_length=args.context_length,
               device=device
           )
           print(f"[Generated at iter {it}, temperature {temperature}, top_p {top_p}]: {full_sentence}")


           # 保存模型检查点
           ckpt_path = os.path.join(args.out_dir, f"ckpt_iter_{it}.pt")
           save_checkpoint(model, optimizer, it, ckpt_path)

       # --------------------------------------------
       # 训练一步
       logits = model(X)
       loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
       optimizer.zero_grad(set_to_none=True) # 梯度清零
       loss.backward() # 反向传播，计算梯度
       grad_norm = gradient_clipping(model.parameters(), args.max_norm) # 梯度裁剪
       optimizer.step() # 更新参数


       '''
       1. 为什么要清零梯度？
       我们用更简单二次方程举例，假设方程fx在x=5的时候一阶导数为零，并且x左侧为减函数，右侧为增函数，x=5就为某区间的最低点（也就是loss的最低点）
       当x=0时计算梯度方向得出变大1的结论（梯度方向为增大，梯度大小为1），参数x变为1，基于这次梯度计算的参数已经更新，所以梯度已经无用所以先清空重新计算
       如果是累加计算实际上是一种针对显存不足的妥协方案，也就是x在某个位置的时候先计算出之后好几步，然后一次性更新（计算平均梯度后一次性更新）
       这种方式可能会走偏（有两个缓解因素：1. 数据独立同分布假设，batch 之间的数据分布相似 2. 学习率调整）
       '''
       # --------------------------------------------


       # 获取下一个批次数据
       X, Y = get_batch(train_data, args.batch_size, args.context_length, device)


       # 每隔log_interval步打印训练进度
       if it % args.log_interval == 0:
           t1 = time.time()
           dt = t1 - t0
           t0 = t1
           print(f"iter {it}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, grad_norm {grad_norm:.4f}")


   # 最终保存（不需要，因为循环的最后一步已经保存了）
   # save_checkpoint(model, optimizer, args.max_iters, os.path.join(args.out_dir, "final_model.pt"))


if __name__ == "__main__":
   """
   程序入口点
  
   当直接运行此脚本时，执行main函数
   """
   main()
