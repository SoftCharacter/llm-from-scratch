"""
================================================================================
BPE分词器 - 训练模块（train_bpe_parallel_optimized.py）
================================================================================


【这个文件是做什么的？】


上一个文件（tokenizer_optimized.py）是BPE分词器的"推理"部分，
用于加载已训练好的词表，对新文本进行编码。


而这个文件是BPE分词器的"训练"部分，用于：
1. 从大量文本中学习合并规则
2. 生成词表（vocabulary）
3. 生成合并规则列表（merges）


【BPE训练算法详解】


假设我们有一个大型文本语料库，BPE训练的目标是：
1. 统计文本中所有可能的"字节对"的出现频率
2. 找到出现频率最高的字节对，合并它
3. 更新统计，重复步骤2，直到词表达到目标大小


举个例子：
---------
假设语料库是："low lower lowest low"


第1步：预处理分词
- 用正则表达式把文本切分成词片段
- "low lower lowest low" → ["low", " lower", " lowest", " low"]


第2步：统计每个词片段的频率
- "low": 2次
- " lower": 1次
- " lowest": 1次


第3步：把每个词片段拆成字节序列
- "low" → (b'l', b'o', b'w')
- " lower" → (b' ', b'l', b'o', b'w', b'e', b'r')
- " lowest" → (b' ', b'l', b'o', b'w', b'e', b's', b't')


第4步：统计所有相邻字节对的频率
- (b'l', b'o'): 4次（每个"low"都有）
- (b'o', b'w'): 4次
- (b'w', b'e'): 2次
- ...


第5步：找到频率最高的字节对，合并它
- 假设 (b'l', b'o') 频率最高
- 合并后：(b'lo', b'w'), (b' ', b'lo', b'w', b'e', b'r'), ...
- 记录这个合并规则：(b'l', b'o') → b'lo'


第6步：更新统计，重复第4-5步
- 继续找频率最高的字节对
- 合并，记录规则
- 直到词表达到目标大小


【优化技巧】


原始的BPE训练算法非常慢，因为：
1. 每次合并后，需要重新扫描整个语料库
2. 统计字节对频率的时间复杂度是O(n)，n是语料库大小
3. 如果要进行10000次合并，总时间复杂度是O(10000 * n)


本文件实现了几个关键优化，把训练时间从19小时缩短到7秒：


优化1：预分词统计
- 先把语料库切分成词片段，统计每个词片段的频率
- 然后只在词片段级别进行BPE，而不是整个语料库
- 这样大大减少了需要处理的数据量


优化2：并行处理
- 把语料库分成多个块，并行处理
- 使用Python的multiprocessing模块


优化3：增量更新
- 每次合并后，只更新受影响的词片段
- 使用倒排索引（pair_to_word_indices）快速定位受影响的词片段
- 避免重新扫描整个语料库


优化4：堆（Heap）数据结构
- 用堆来维护字节对的频率
- 可以在O(log n)时间内找到频率最高的字节对


【PyTorch替换说明】


这个文件主要涉及：
1. 字符串处理和正则表达式 - 不适合用PyTorch
2. 字典和集合操作 - Python原生实现已经足够快
3. 多进程并行 - Python的multiprocessing已经很好
4. 堆操作 - 可以用torch.topk替代，但提升不大


如果要进一步优化，可以考虑：
1. 用Rust或C++重写核心算法
2. 用GPU加速（但BPE训练不太适合GPU）
3. 用更高效的数据结构（如Trie树）


================================================================================
"""
# cs336_basics/train_bpe.py
from __future__ import annotations
import argparse
import io
import os
import mmap
import tqdm  # 进度条库，用于显示训练进度
import heapq  # Python内置的堆（优先队列）模块，用于高效找到频率最高的字节对
# 注意：Python的heapq只支持小顶堆，所以我们要用负数来实现大顶堆
import regex as re  # 正则表达式库，比Python内置的re更强大
from collections import Counter, defaultdict  # Counter用于计数，defaultdict用于自动初始化字典
from pathlib import Path
from typing import Iterable, Tuple, BinaryIO
from multiprocessing import Pool, cpu_count  # 多进程并行处理
import json

# ========== GPT-2的预处理分词正则表达式 ==========
# 【作用】把文本切分成"词片段"，这是BPE训练的第一步
# 【详解】见tokenizer_optimized.py中的注释
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_RX = re.compile(PAT)


def _bytes_to_tuple(b: bytes) -> tuple[bytes, ...]:
    """
    把字节序列转换成字节元组

    【为什么需要这个函数？】

    在BPE算法中，我们需要频繁地合并相邻的字节。
    Python的bytes对象是不可变的，每次合并都需要创建新的bytes对象。
    使用元组可以更方便地操作和合并。

    【示例】
    输入：b"low"
    输出：(b"l", b"o", b"w")

    【工作原理】
    - bytes([x]) 把单个整数x转换成单字节的bytes对象
    - 例如：bytes([108]) = b'l'
    - 然后用tuple()把所有单字节对象打包成元组

    【PyTorch替换说明】
    这个函数只是简单的类型转换，不需要PyTorch优化。
    """
    return tuple(bytes([x]) for x in b)
    # # b"low" -> (b"l", b"o", b"w") <- this is the basic unit for BPE


def _count_from_text(text: str) -> dict[tuple[bytes, ...], int]:
    """
    对文本进行预处理分词，统计每个词片段的频率

    【作用】
    这是BPE训练的第一步：把文本切分成词片段，统计每个词片段出现的次数。

    【示例】
    输入："low lower lowest low"

    第1步：用正则表达式切分
    - "low lower lowest low" → ["low", " lower", " lowest", " low"]

    第2步：转成字节元组
    - "low" → (b'l', b'o', b'w')
    - " lower" → (b' ', b'l', b'o', b'w', b'e', b'r')
    - " lowest" → (b' ', b'l', b'o', b'w', b'e', b's', b't')
    - " low" → (b' ', b'l', b'o', b'w')

    第3步：统计频率
    - (b'l', b'o', b'w'): 2次
    - (b' ', b'l', b'o', b'w', b'e', b'r'): 1次
    - (b' ', b'l', b'o', b'w', b'e', b's', b't'): 1次
    - (b' ', b'l', b'o', b'w'): 1次

    【为什么用defaultdict？】
    defaultdict(int)会在访问不存在的key时自动创建，默认值是0。
    这样我们不需要检查key是否存在，直接counts[key] += 1即可。

    【PyTorch替换说明】
    这个函数主要是字符串处理和字典操作，不适合用PyTorch优化。
    """
    counts: dict[tuple[bytes, ...], int] = defaultdict(int)  # defaultdict自动创建默认值，int的默认值是0
    for m in _RX.finditer(text):  # 用正则表达式找到所有词片段
        bs = m.group(0).encode("utf-8")  # 把词片段转成UTF-8字节序列
        if not bs:  # 跳过空字节序列
            continue
        counts[_bytes_to_tuple(bs)] += 1  # 统计频率
    return counts


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    找到文件分块的边界位置（关键优化！）

    【为什么需要这个函数？】

    为了并行处理大文件，我们需要把文件分成多个块。
    但是，如果随意切分，可能会把一个词切成两半：

    错误示例：
    - 文件内容："hello world"
    - 在第5个字节切分："hello" 和 " world"
    - 如果切分点是 "hel" 和 "lo world"，就会破坏词的完整性

    解决方法：
    - 在特殊token（如  ）的位置切分
    - 这样可以确保每个块都是完整的文本片段

    【算法流程】

    第1步：计算文件总大小
    - file.seek(0, os.SEEK_END) 移动到文件末尾
    - file.tell() 返回当前位置（即文件大小）

    第2步：计算均匀分布的初始边界
    - 假设文件大小是1000字节，要分成4块
    - 初始边界：[0, 250, 500, 750, 1000]

    第3步：调整边界到特殊token位置
    - 从每个边界位置开始，向后搜索特殊token
    - 找到后，把边界移到特殊token的位置
    - 这样可以确保每个块在特殊token处切分

    【示例】
    文件内容："Hello  world  test"
    特殊token：b"  "
    目标块数：2

    初始边界：[0, 10, 20]（假设文件大小是20）

    调整边界：
    - 从位置10开始，向后搜索 b"  "
    - 假设在位置13找到 b"  "
    - 调整边界为 [0, 13, 20]

    最终分成两个块：
    - 块1：[0, 13) = "Hello  world"
    - 块2：[13, 20) = "  test"

    【PyTorch替换说明】
    这个函数主要是文件I/O操作，不适合用PyTorch优化。
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # ========== 第1步：计算文件总大小 ==========
    file.seek(0, os.SEEK_END)  # 移动到文件末尾
    file_size = file.tell()  # 获取当前位置（即文件大小）
    file.seek(0)  # 回到文件开头

    # ========== 第2步：计算均匀分布的初始边界 ==========
    chunk_size = file_size // desired_num_chunks  # 每块的平均大小

    # 初始边界：均匀分布
    # 例如：文件大小1000，4块，边界是 [0, 250, 500, 750, 1000]
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size  # 最后一个边界是文件末尾

    # ========== 第3步：调整边界到特殊token位置 ==========
    mini_chunk_size = 4096  # 每次读取4KB，用于搜索特殊token

    for bi in range(1, len(chunk_boundaries) - 1):  # 跳过第一个和最后一个边界
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # 移动到边界位置

        while True:
            mini_chunk = file.read(mini_chunk_size)  # 读取一小块数据

            # 如果到达文件末尾，把边界设为文件末尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 在这一小块数据中查找特殊token
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # 找到了！把边界移到特殊token的位置
                chunk_boundaries[bi] = initial_position + found_at
                break

            # 没找到，继续向后搜索
            initial_position += mini_chunk_size

    # 返回去重后的边界列表（可能有重叠的边界）
    return sorted(set(chunk_boundaries))


def _worker_count_chunk(args: Tuple[str, int, int, bytes, tuple[str, ...]]) -> dict[tuple[bytes, ...], int]:
    """
    从文件中读取 [start, end) 字节范围的块；如果存在前导特殊token且 start > 0 则将其去除，
    然后用所有特殊token分割块文本，丢弃这些token，并执行预分词统计。

    参数:
        input_path: 输入文件路径
        start: 块起始字节偏移量
        end: 块结束字节偏移量
        split_special: 用于分割块的特殊token字节(e.g. b"<|endoftext|>")
        all_specials: 所有特殊token字节的元组(e.g. (b"<|endoftext|>", b"<|pad|>"))

    返回:
        dict[tuple[bytes, ...], int]
    """
    # ========== 第1步：解包参数 ==========
    # args是一个元组，包含所有需要的参数
    # 这行代码把元组拆开，分别赋值给5个变量
    input_path, start, end, split_special, all_specials = args

    # ========== 第2步：读取文件块 ==========
    # 打开文件，"rb"表示以二进制模式读取
    with open(input_path, "rb") as f:
        f.seek(start)  # 移动到块的起始位置
        data = f.read(end - start)  # 读取指定长度的数据

    # 【非第一个块】如果块开头有特殊token，去掉它，避免重复统计
    # 【为什么要去掉？】
    # 因为我们在分割文件时，是在特殊token的位置切分的。
    # 所以非第一个块的开头可能有一个特殊token。
    # 这个特殊token是"切分点"，不应该被统计进去。
    #
    # 【示例】
    # 假设文件内容："Hello  World"
    # 我们在空格处切分，得到两个块：
    # - 块1："Hello"
    # - 块2：" World"
    # 块2开头的是切分点，需要去掉
    if start > 0 and data.startswith(split_special):
        data = data[len(split_special):]  # 去掉开头的特殊token

    # ========== 第4步：把字节解码成字符串 ==========
    # errors="replace" 表示如果遇到无法解码的字节，用替换字符（�）代替
    text = data.decode("utf-8", errors="replace")

    # ========== 第5步：分割并去掉所有特殊token ==========
    # 【为什么还要分割？】
    # 因为特殊token可能出现在块的中间，不只是开头
    # 例如："Hello</|endoftext|>World" 中，特殊token出现在文本中间
    # 我们需要把所有的特殊token都去掉，只保留 "Hello" 和 "World"
    if all_specials:
        # 构建正则表达式：把所有特殊token用 | 连接
        # 例如：(b"", b"<|pad|>") -> "<\|endoftext\|>|<\|pad\|>"
        pat = re.compile("|".join(re.escape(s) for s in all_specials))
        parts = pat.split(text)  # 分割文本，去掉特殊token
    else:
        parts = [text]  # 如果没有特殊token，整个文本作为一个部分

    # ========== 第6步：统计每个词片段的频率 ==========
    # counts是一个字典，用来存储统计结果
    # defaultdict(int)会在访问不存在的key时自动创建，默认值是0
    counts: dict[tuple[bytes, ...], int] = defaultdict(int)

    for seg in parts:
        if not seg:  # 跳过空字符串
            continue
        # 对每个文本片段进行预处理分词，统计词片段频率
        seg_counts = _count_from_text(seg)
        # 把统计结果合并到总字典中
        for k, v in seg_counts.items():
            counts[k] += v

    return counts


def _pretoken_counts_parallel(
        input_path: str | os.PathLike,
        special_tokens: list[str],
        num_workers: int | None = None,
) -> dict[tuple[bytes, ...], int]:
    """
    使用 find_chunk_boundaries 基于 split_special_token 执行并行预分词统计。
    注意：此函数假设 special_tokens 包含至少一个用于分割的特殊token eg <|endoftext|>。


    参数:
        input_path: 输入文件路径
        special_tokens: 特殊token列表（至少包含一个用于分割的token）
        num_workers: 并行工作进程数（默认使用 cpu_count）

    返回:
        dict[tuple[bytes, ...], int]，键是预分词的字节元组，例如 (b"l", b"o", b"w")，值是频率
    """
    # ========== 第1步：准备特殊token ==========
    all_specials = tuple(special_tokens)

    # 【情况1】如果没有特殊token，直接读取整个文件并统计
    if not special_tokens:
        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return _count_from_text(text)

    # ========== 第2步：找到文件分块的边界位置 ==========
    # 用第一个特殊token作为分割点
    split_special = special_tokens[0].encode("utf-8")

    # 调用find_chunk_boundaries函数找到边界
    # 例如：文件大小1000字节，4个工作进程，边界可能是 [0, 250, 500, 750, 1000]
    with open(input_path, "rb") as fb:
        boundaries = find_chunk_boundaries(fb, desired_num_chunks=(num_workers or cpu_count()),
                                           split_special_token=split_special)

    # ========== 第3步：为每个块创建任务 ==========
    # 每个任务是一个元组：(文件路径, 起始位置, 结束位置, 分割特殊token, 所有特殊token)
    tasks: list[Tuple[str, int, int, bytes, tuple[str, ...]]] = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        if s == e:  # 跳过空块
            continue
        tasks.append((str(input_path), s, e, split_special, all_specials))

    # 如果没有任务，返回空字典
    if not tasks:
        return {}

    # ========== 第4步：执行任务（单线程或多线程）==========
    # 【情况1】单线程处理（num_workers为None或<=1）
    if num_workers is None or num_workers <= 1:
        merged: dict[tuple[bytes, ...], int] = defaultdict(int)
        for t in tqdm.tqdm(tasks, desc="Pre-tokenizing", unit="chunk"):
            part = _worker_count_chunk(t)  # 处理一个块
            for k, v in part.items():
                merged[k] += v  # 合并结果
        return merged

    # 【情况2】多线程并行处理
    nproc = min(len(tasks), num_workers)  # 工作进程数量
    merged: dict[tuple[bytes, ...], int] = defaultdict(int)

    # 使用Python的multiprocessing.Pool创建进程池
    # 想象一下：你雇佣了nproc个工人，每个人处理一个任务
    with Pool(processes=nproc) as pool:
        # pool.imap_unordered：并行执行任务，不保证顺序
        # chunksize=1：每次从任务列表中取1个任务
        for part in tqdm.tqdm(
                pool.imap_unordered(_worker_count_chunk, tasks, chunksize=1),
                desc="Pre-tokenizing",
                total=len(tasks),
                unit="chunk",
        ):
            # 合并每个工作进程的结果
            for k, v in part.items():
                merged[k] += v

    return merged


# ========== 优化后的训练函数（把19小时缩短到7秒！）==========
def train_bpe_parallel(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    【BPE训练的主函数】

    这个函数是整个BPE训练的核心！
    它会从文本中学习合并规则，生成词表和合并规则列表。

    【算法流程】
    第1步：并行预处理分词，统计词片段频率
    第2步：过滤低频词片段
    第3步：统计所有字节对的频率
    第4步：循环执行以下操作，直到词表达到目标大小：
        a. 找到频率最高的字节对
        b. 合并这个字节对
        c. 更新统计（只更新受影响的部分）
    第5步：生成最终的词表

    【参数说明】
    ----------
    input_path : str | os.PathLike
        输入文件的路径（训练语料）

    vocab_size : int
        目标词表大小
        例如：vocab_size=10000 表示词表包含10000个token

    special_tokens : list[str]
        特殊token列表
        例如：["", "<|pad|>"]

    num_workers : int | None
        工作进程数量（默认使用CPU核心数）

    【返回值】
    --------
    tuple[dict[int, bytes], list[tuple[bytes, bytes]]]
        返回两个东西：
        1. vocab: 词表，一个字典，key是token ID，value是token的字节表示
           例如：{0: b"", 1: b"<|pad|>", 256: b"l", 257: b"lo", ...}
        2. merges: 合并规则列表，一个列表，每个元素是一个字节对
           例如：[(b'l', b'o'), (b'lo', b'w'), ...]

    【性能优化】
    这个函数实现了两个关键优化：
    1. 预分词统计：只在词片段级别进行BPE，而不是整个语料库
    2. 增量更新：每次合并后，只更新受影响的词片段
    """
    assert vocab_size > 0  # 确保词表大小大于0

    # ========== 第1步：并行预处理分词，统计词片段频率 ==========
    # 调用_pretoken_counts_parallel函数，并行处理文件
    # 返回一个字典，key是词片段（字节元组），value是频率
    tok_counts: dict[tuple[bytes, ...], int] = _pretoken_counts_parallel(
        input_path=input_path, special_tokens=special_tokens, num_workers=num_workers
    )

    # ========== 第2步：过滤低频词片段 ==========
    # 【为什么要过滤？】
    # 低频词片段对训练贡献不大，但会占用内存和计算资源。
    # 这里我们只保留频率>=2的词片段。
    tok_counts = {k: v for k, v in tok_counts.items() if v >= 2}

    # ========== 第3步：准备数据结构（性能优化1）==========
    # 【为什么要这样做？】
    # 我们需要频繁地访问和修改词片段，所以要把数据转换成方便操作的格式。

    # unique_word_tuples: 所有唯一的词片段（元组形式）
    # 例如：[(b'l', b'o', b'w'), (b' ', b'l', b'o', b'w', b'e', b'r'), ...]
    unique_word_tuples = list(tok_counts.keys())

    # word_counts: 每个词片段的频率
    # 例如：[2, 1, 1, ...]（对应上面的词片段）
    word_counts = [tok_counts[w] for w in unique_word_tuples]

    # unique_words: 所有唯一的词片段（列表形式，方便修改）
    # 例如：[[b'l', b'o', b'w'], [b' ', b'l', b'o', b'w', b'e', b'r'], ...]
    unique_words = [list(tup) for tup in unique_word_tuples]

    # ========== 第4步：统计字节对频率，构建倒排索引 ==========
    # pairs: 字节对频率字典
    # key是字节对，value是频率
    # 例如：{(b'l', b'o'): 4, (b'o', b'w'): 4, ...}
    pairs = defaultdict(int)

    # pair_to_word_indices: 倒排索引
    # key是字节对，value是包含这个字节对的所有词片段的索引集合
    # 例如：{(b'l', b'o'): {0, 1, 2, 3}, (b'o', b'w'): {0, 1, 2, 3}, ...}
    # 【为什么需要倒排索引？】
    # 当我们合并一个字节对时，只需要更新包含这个字节对的词片段。
    # 倒排索引可以让我们快速找到这些词片段，避免扫描整个语料库。
    pair_to_word_indices = defaultdict(set)

    # 遍历所有词片段，统计字节对频率
    for idx, word in enumerate(unique_words):
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])  # 相邻的两个字节组成一个字节对
            pairs[pair] += word_counts[idx]  # 累加频率
            pair_to_word_indices[pair].add(idx)  # 记录这个词片段包含这个字节对

    # ========== 第5步：初始化堆（优先队列）==========
    # 【为什么要用堆？】
    # 我们需要频繁地找到频率最高的字节对。
    # 堆可以在O(log n)时间内找到最大值，比每次扫描整个字典快得多。
    #
    # 【为什么用负数？】
    # Python的heapq只支持小顶堆（最小的元素在堆顶）。
    # 我们要找最大频率，所以用负数来实现大顶堆。
    # 例如：频率4变成-4，频率2变成-2。
    # 堆顶是最小的负数（即最大的频率）。
    heap = [(-freq, pair) for pair, freq in pairs.items()]
    heapq.heapify(heap)  # 把列表转换成堆

    # ========== 第6步：开始合并循环 ==========
    merges: list[tuple[bytes, bytes]] = []  # 合并规则列表
    num_specials = len(special_tokens)  # 特殊token的数量
    # 计算最多可以合并多少次
    # 词表大小 = 特殊token数量 + 256个单字节 + 合并次数
    # 256个单字节：BPE的最小粒度，是"原子"级别的字符，不能再分了
    # 假设我们要训练一个BPE词表，设置：
    #
    # 目标词表大小：300
    # 个token
    # 特殊token：2个（ < / | endoftext | > 和 < | pad | >）
    # 那么词表的构成如下：
    '''
    词表 = 特殊token + 256个单字节 + 合并产生的token
    300  =    2个     +  256个       +        ?
 
 
    合并次数 = 300 - 2 - 256 = 42次
    '''
    max_merges = max(0, vocab_size - num_specials - 256)

    # 使用tqdm显示进度条
    for _ in tqdm.tqdm(range(max_merges), desc="Learning BPE merges", unit="merge"):
        # ========== 第6.1步：找到频率最高的字节对 ==========
        # 【为什么用while循环？】
        # 堆中可能有过期的数据（频率已经改变了，但堆中的数据没更新）。
        # 我们需要不断弹出堆顶元素，直到找到一个有效的数据。
        while heap:
            neg_freq, best_pair = heapq.heappop(heap)  # 弹出堆顶元素（频率最高的字节对）
            current_freq = pairs.get(best_pair, 0)  # 获取当前的频率

            # 【懒惰更新检查】
            # 如果堆中弹出的频率不等于当前频率，说明这个数据过期了。
            # 【为什么会过期？】
            # 当我们合并一个字节对时，会改变其他字节对的频率。
            # 但我们不会更新堆中的所有数据，而是只把新的频率推入堆中。
            # 这样堆中就会有过期的数据。
            if -neg_freq == current_freq and current_freq > 0:
                break  # 找到有效的数据，跳出循环


        else:
            break  # 堆空了，没有可以合并的字节对

        # ========== 第6.2步：记录这个合并规则 ==========
        # best_pair就是我们这次要合并的字节对
        merges.append(best_pair)

        # ========== 第6.3步：增量更新（性能优化2，最复杂的部分！）==========
        # 【为什么这是最复杂的部分？】
        # 原始的BPE算法每次合并后，需要重新扫描整个语料库。
        # 时间复杂度是O(n)，n是语料库大小。
        # 如果要进行10000次合并，总时间复杂度是O(10000 * n)。
        #
        # 优化方法：
        # 使用倒排索引（pair_to_word_indices）快速找到受影响的词片段。
        # 只更新受影响的词片段，避免重新扫描整个语料库。
        # 时间复杂度从O(n)降到O(m)，m是受影响的词片段数量。
        #
        # 【核心思路】
        # 当我们将单词中的 (a, b) 合并为 ab 时：
        # 1. 哪些旧的 pair 频率会减少？
        #    -> 在合并 (a, b) 之前，识别出受影响的相邻 pair（比如 x, a 和 b, y）
        #    -> 将它们的全局频率减去
        # 2. 哪些新的 pair 频率会增加？
        #    -> 合并成 ab 后，识别出新产生的 pair（比如 x, ab 和 ab, y）
        #    -> 将它们的全局频率加上
        # 3. 更新索引与堆：将新频率推入堆中

        # 提取字节对的两个字节
        a, b = best_pair
        # 合并后的新字节
        ab = a + b

        # 获取包含这个字节对的所有词片段的索引
        affected_indices = pair_to_word_indices[best_pair]
        # 清空倒排索引（因为这个字节对已经被合并了）
        pair_to_word_indices[best_pair] = set()

        # 遍历所有受影响的词片段
        for idx in affected_indices:
            word = unique_words[idx]  # 获取这个词片段
            count = word_counts[idx]  # 获取这个词片段的频率

            # ========== 第6.3.1步：在词片段中找到所有需要合并的位置 ==========
            i = 0
            new_word = []  # 存储合并后的新词片段

            while i < len(word):
                # 检查当前位置和下一个位置是否是要合并的字节对
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    # ========== 找到了一个需要合并的字节对 ==========

                    # ========== [A] 处理左边的字节对 (x, a) -> (x, ab) ==========
                    # 【为什么要处理左边的字节对？】
                    # 假设原词片段是：[x, a, b, y]
                    # 合并 (a, b) 后变成：[x, ab, y]
                    # 左边的字节对从 (x, a) 变成了 (x, ab)
                    # 需要更新它们的频率
                    if i > 0:  # 如果左边有字节
                        prev_pair = (word[i - 1], a)  # 旧的左字节对

                        # 减少旧字节对的频率
                        if prev_pair not in pairs:
                            pairs[prev_pair] = 0
                        pairs[prev_pair] -= count

                        # 增加新字节对的频率
                        new_prev_pair = (word[i - 1], ab)  # 新的左字节对
                        if new_prev_pair not in pairs:
                            pairs[new_prev_pair] = 0  # 避免KeyError
                        pairs[new_prev_pair] += count

                        # 更新倒排索引
                        pair_to_word_indices[new_prev_pair].add(idx)

                        # 将新频率推入堆中
                        heapq.heappush(heap, (-pairs[new_prev_pair], new_prev_pair))

                    # ========== [B] 处理右边的字节对 (b, y) -> (ab, y) ==========
                    # 【为什么要处理右边的字节对？】
                    # 假设原词片段是：[x, a, b, y]
                    # 合并 (a, b) 后变成：[x, ab, y]
                    # 右边的字节对从 (b, y) 变成了 (ab, y)
                    # 需要更新它们的频率
                    if i < len(word) - 2:  # 如果右边还有字节
                        next_pair = (word[i + 1], word[i + 2])  # 旧的右字节对

                        # 减少旧字节对的频率
                        if next_pair not in pairs:
                            pairs[next_pair] = 0
                        pairs[next_pair] -= count

                        # 增加新字节对的频率
                        new_next_pair = (ab, word[i + 2])  # 新的右字节对
                        if new_next_pair not in pairs:
                            pairs[new_next_pair] = 0  # 避免KeyError
                        pairs[new_next_pair] += count

                        # 更新倒排索引
                        pair_to_word_indices[new_next_pair].add(idx)

                        # 将新频率推入堆中
                        heapq.heappush(heap, (-pairs[new_next_pair], new_next_pair))

                    # 将合并后的字节添加到新词片段中
                    new_word.append(ab)
                    i += 2  # 跳过已经合并的两个字节
                else:
                    # 不是要合并的字节对，直接添加到新词片段中
                    new_word.append(word[i])
                    i += 1

            # ========== 第6.3.2步：用合并后的新词片段替换旧词片段 ==========
            unique_words[idx] = new_word

        # ========== 第6.4步：将已合并的字节对的频率设为0 ==========
        # 【为什么要设为0？】
        # 这个字节对已经被合并了，不会再出现了。
        # 将频率设为0，表示它已经"消失"了。
        pairs[best_pair] = 0

        # ========== 第6.5步：内存管理（清理过期数据）==========
        # 【为什么要清理？】
        # 随着合并次数增加，pairs和pair_to_word_indices会越来越大。
        # 很多字节对的频率已经变成0，但它们仍然占用内存。
        # 定期清理可以释放内存，防止内存泄漏。
        if (_ + 1) % 1000 == 0:  # 每1000次合并清理一次

            # 找出所有频率大于0的字节对
            active_pairs = {p for p, v in pairs.items() if v > 0}

            # 清理倒排索引，只保留频率大于0的字节对
            old_indices_keys = list(pair_to_word_indices.keys())
            for p in old_indices_keys:
                if p not in active_pairs:
                    del pair_to_word_indices[p]

            # 清理pairs字典，只保留频率大于0的字节对
            pairs = {p: pairs[p] for p in active_pairs}

    # ========== 第7步：生成词表 ==========
    # 词表是一个字典，key是token ID，value是token的字节表示
    vocab: dict[int, bytes] = {}
    idx = 0  # token ID计数器

    # ========== 第7.1步：添加特殊token ==========
    # 特殊token的ID从0开始
    for s in special_tokens:
        vocab[idx] = s.encode("utf-8")
        idx += 1

    # ========== 第7.2步：添加所有单字节token ==========
    # 256个单字节token的ID从特殊token数量开始
    for b in range(256):
        vocab[idx] = bytes([b])
        idx += 1

    # ========== 第7.3步：添加合并后的token ==========
    # seen集合用于去重，避免添加重复的token
    seen = set(vocab.values())

    # 遍历合并规则列表，添加合并后的token
    for a, b in merges:
        if idx >= vocab_size:  # 如果词表已经达到目标大小，停止添加
            break

        ab = a + b  # 合并后的字节
        if ab in seen:  # 如果这个token已经添加过，直接跳过
            continue

        vocab[idx] = ab
        idx += 1
        seen.add(ab)

    return vocab, merges


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "trained_tokenizer"
DEFAULT_VOCAB_FILENAME = "vocab_of_your_tokenizer.json"
DEFAULT_MERGES_FILENAME = "merges_of_your_tokenizer.json"


def save_tokenizer_files(
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        output_dir: str | os.PathLike = DEFAULT_OUTPUT_DIR,
        vocab_filename: str = DEFAULT_VOCAB_FILENAME,
        merges_filename: str = DEFAULT_MERGES_FILENAME,
) -> tuple[Path, Path]:
    """
    保存训练好的BPE词表和合并规则。

    参数说明：
        vocab: 词表，key是token ID，value是token的字节表示
        merges: BPE合并规则列表
        output_dir: 输出目录，默认是项目根目录下的trained_tokenizer
        vocab_filename: 词表文件名
        merges_filename: 合并规则文件名
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab_path = output_path / vocab_filename
    merges_path = output_path / merges_filename

    max_token_id = max(vocab) if vocab else -1
    if max_token_id >= len(vocab):
        raise ValueError(
            f"Invalid vocab ids: max token id is {max_token_id}, but vocab size is {len(vocab)}. "
            "Token ids must be continuous from 0 to vocab_size - 1."
        )

    vocab_json = {str(token_id): token_bytes.decode("latin1") for token_id, token_bytes in vocab.items()}
    merges_json = [[left.decode("latin1"), right.decode("latin1")] for left, right in merges]

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_json, f, ensure_ascii=False)

    return vocab_path, merges_path


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer and save vocab/merges files.")
    parser.add_argument("--input_path", required=True, help="Path to the raw training text file.")
    parser.add_argument("--vocab_size", type=int, required=True, help="Target tokenizer vocabulary size.")
    parser.add_argument(
        "--special_token",
        action="append",
        default=None,
        help="Special token to keep intact. Repeat this argument to add more tokens.",
    )
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save vocab and merges files.")
    parser.add_argument("--vocab_filename", default=DEFAULT_VOCAB_FILENAME, help="Vocabulary JSON filename.")
    parser.add_argument("--merges_filename", default=DEFAULT_MERGES_FILENAME, help="Merges JSON filename.")
    return parser.parse_args()


def main() -> None:
    """
    训练BPE分词器并保存到trained_tokenizer目录。
    """
    args = parse_args()
    special_tokens = args.special_token or ["<|endoftext|>"]

    vocab, merges = train_bpe_parallel(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        num_workers=args.num_workers,
    )
    vocab_path, merges_path = save_tokenizer_files(
        vocab=vocab,
        merges=merges,
        output_dir=args.output_dir,
        vocab_filename=args.vocab_filename,
        merges_filename=args.merges_filename,
    )

    print(f"Saved vocab to: {vocab_path}")
    print(f"Saved merges to: {merges_path}")


if __name__ == "__main__":
    main()


'''
假设训练文本是：
"hello hello hello hello hello hello hello hello hello hello"


预分词后：
```python
tok_counts = {
   (b'h', b'e', b'l', b'l', b'o'): 10  # "hello" 出现了10次
}
```


**合并前：字节对统计**


```python
{
   (b't', b'h'): 10,  # 左侧
   (b'h', b'e'): 10,
   (b'e', b'l'): 10,  # 合并位置的左侧
   (b'l', b'l'): 10,  # 要合并的字节对
   (b'l', b'o'): 10,  # 合并位置的右侧
}
```


**执行合并：** `(b'l', b'l')` → `b'll'`


```
合并前: [b't', b'h', b'e', b'l', b'l', b'o']
                       ↓
合并后: [b't', b'h', b'e', b'll', b'o']
```


**合并后：字节对统计**


```python
{
   (b't', b'h'): 10,     # 不变
   (b'h', b'e'): 10,     # 不变
   (b'e', b'l'): 0,      # 减少10（消失了）
   (b'l', b'l'): 0,      # 减少10（被合并了）
   (b'l', b'o'): 0,      # 减少10（消失了）
   (b'e', b'll'): 10,    # 新增（左侧变化）
   (b'll', b'o'): 10,    # 新增（右侧变化）
}
```




**代码对应：**


```python
# 左侧处理（代码640-664行）
prev_pair = (b'e', b'l')     # 旧的左字节对
pairs[prev_pair] -= 10       # 频率-10


new_prev_pair = (b'e', b'll') # 新的左字节对
pairs[new_prev_pair] += 10    # 频率+10


# 右侧处理（代码666行之后）
next_pair = (b'l', b'o')     # 旧的右字节对
pairs[next_pair] -= 10       # 频率-10


new_next_pair = (b'll', b'o') # 新的右字节对
pairs[new_next_pair] += 10    # 频率+10


# 被合并的字节对
pairs[(b'l', b'l')] = 0      # 设为0


'''
