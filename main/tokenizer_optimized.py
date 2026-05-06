"""
An optimized Byte Pair Encoding (BPE) tokenizer implementation.
This is the same as the tokenizer_optimized.py in the `https://github.com/Siyuan-Harry/bpe-optimized-from-scratch` repo.
I include this into the project just to avoid dependency warning on `run_train_model.py`.
"""
"""
================================================================================
BPE分词器 - 推理模块（tokenizer_optimized.py）
================================================================================


【什么是BPE？- 给小白的通俗解释】


BPE（Byte Pair Encoding，字节对编码）是一种文本分词算法。
它的核心思想非常简单：把文本中经常一起出现的字符"合并"成一个整体。


举个例子：
---------
假设我们有一段文本："low lower lowest"


第一步，统计所有相邻字符对的出现频率：
 - "lo" 出现 3 次
 - "ow" 出现 3 次
 - "we" 出现 2 次
 - "er" 出现 2 次
 - ...


第二步，找到出现频率最高的字符对，比如 "lo"，把它合并成一个新符号：
 - 原来：l o w   l o w e r   l o w e s t
 - 合并后：low   lower   lowest
 - 现在我们有了一个新的"词元"（token）："lo"


第三步，重复这个过程，每次合并频率最高的字符对：
 - 再合并 "ow" → "low"
 - 再合并 "er" → "er"
 - ...


最终，我们得到一个"词表"（vocabulary），里面包含了：
 - 单个字符：a, b, c, ..., z, 0, 1, ..., 9, ...
 - 合并后的字符对：th, he, in, er, ...
 - 更长的组合：the, ing, tion, ...


【为什么要用BPE？】


传统分词方法有两种极端：
1. 按字符分词：词表小（只有256个字节），但序列很长
  - "hello" → ['h', 'e', 'l', 'l', 'o']（5个token）

2. 按词分词：序列短，但词表巨大（可能上百万）
  - "hello" → ['hello']（1个token）
  - 问题：遇到新词就懵了，比如"unfriendliness"可能不在词表中


BPE是折中方案：
- 常见的词整体作为一个token（效率高）
- 不常见的词拆成子词或字符（泛化能力强）
- 词表大小可控（通常几万到几十万）
- 能够处理多语言文本


【GPT系列模型使用的BPE】


GPT-2/GPT-3/GPT-4 都使用字节级BPE：
- 先把文本转成字节（UTF-8编码）
- 然后在字节级别进行BPE合并
- 这样可以处理任何语言的文本，不会遇到"未知字符"


【本文件的作用】


这个文件实现了BPE分词器的"推理"部分：
- 加载已经训练好的词表和合并规则
- 把新文本转换成token ID序列
- 把token ID序列还原成文本


优化说明：
   这个实现包含了三个主要优化：
   1. 将合并规则转换为字典，实现O(1)查询
   2. 预构建反向词汇表
   3. 引入缓存机制


【PyTorch替换说明】


这个文件主要是字符串操作，不太涉及PyTorch的核心功能。
但以下地方可以考虑用PyTorch优化（如果需要GPU加速）：
- 缓存机制可以用torch的tensor缓存
- 批量编码时可以用torch并行处理


可以直接使用tiktoken库替换：
   import tiktoken
   tokenizer = tiktoken.get_encoding("gpt2")


实现区别：
   - 功能基本等价，但tiktoken是官方实现，更稳定
   - 这个实现展示了BPE的内部工作原理
================================================================================
"""
import regex as re  # 正则表达式库，比Python内置的re更强大，支持Unicode属性
from typing import Iterable, Iterator, Dict, List, Tuple
import json


class Tokenizer:
    """
    BPE分词器类 - 用于将文本转换为token ID序列，以及反向解码

    【核心概念解释】

    1. vocab（词表 Vocabulary）：所有可能的token的集合。一个字典，key是token ID（整数），value是token对应的字节序列
       例如：{0: b' <|endoftext|>', 1: b'a', 2: b'b', ..., 256: b'th', 257: b'the', ...}

    2. merges（合并规则）：一个列表，记录了BPE训练过程中学到的合并顺序
       例如：[(b't', b'h'), (b'th', b'e'), ...]
       表示：先把 't' 和 'h' 合并成 'th'，再把 'th' 和 'e' 合并成 'the'

    3. special_tokens（特殊token）：一些有特殊含义的token
       例如：' <|endoftext|>' 表示文本结束，'<|pad|>' 表示填充，<eos>（结束符）、<pad>（填充符）等

    【编码过程】（文本 → token ID序列）

    输入："hello world"

    第1步：预处理分词（Pre-tokenization）
    - 使用正则表达式把文本切分成"词片段"
    - 例如："hello world" → ["hello", " world"]

    第2步：对每个词片段进行BPE编码
    - 把词片段转成字节序列：b'hello'
    - 初始化为单字节列表：[b'h', b'e', b'l', b'l', b'o']
    - 根据merges规则，逐步合并相邻的字节对
    - 假设 'll' 是一个合并规则，合并后：[b'h', b'e', b'll', b'o']
    - 最终得到token列表，查词表得到ID：[104, 101, 271, 111]

    第3步：返回token ID序列

    【解码过程】（token ID序列 → 文本）

    输入：[104, 101, 271, 111]

    - 根据词表，把每个ID转回字节序列
    - 拼接所有字节：b'hello'
    - 用UTF-8解码成字符串："hello"
    """

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]],
                 special_tokens: List[str] | None = None):
        """
        初始化BPE分词器

        参数说明：
        ----------
        vocab : Dict[int, bytes]
            词表字典
            - key: token ID（整数，从0开始编号）
            - value: token对应的字节序列
            - 例如：{0: b'a', 1: b'b', 256: b'th', 257: b'the'}
            - 注意：前256个ID（0-255）通常对应256个单字节

        merges : List[Tuple[bytes, bytes]]
            合并规则列表（按优先级排序）
            - 每个元素是一个元组 (字节A, 字节B)，表示把A和B合并
            - 列表的顺序就是合并的优先级顺序
            - 例如：[(b't', b'h'), (b'th', b'e')] 表示先合并 th，再合并 the

        special_tokens : List[str] | None
            特殊token列表（可选）
            - 例如：[' <|endoftext|>', '<|pad|>']
            - 这些token有特殊含义，不会被BPE拆分

        【PyTorch替换说明】
        这个初始化过程主要是数据结构的构建，不涉及计算密集型操作，
        使用PyTorch不会带来明显的性能提升。
        """
        self.vocab = vocab  # 保存词表

        # ========== 优化1：将合并规则转换为字典，实现O(1)查询 ==========
        # 【原理】
        # 原始的merges是一个列表，查找某个合并规则的优先级需要遍历整个列表，时间复杂度O(n)
        # 我们把它转成一个字典，key是合并规则（字节对），value是优先级（排名）
        # 这样查找优先级的时间复杂度变成O(1)，大大加速了编码过程
        # ranks[pair] = i 表示pair在第i步被合并
        # 这样可以快速判断某个字节对是否应该合并

        # 【示例】
        # merges = [(b't', b'h'), (b'th', b'e'), (b'i', b'n')]
        # 转换后：
        # ranks = {(b't', b'h'): 0, (b'th', b'e'): 1, (b'i', b'n'): 2}
        # 排名越小，优先级越高（越先合并）
        #
        # 【PyTorch替换说明】
        # 可以用torch.tensor存储排名，然后用torch.searchsorted进行查找
        # 但对于这种字典查找场景，Python字典的性能已经足够好
        self.ranks = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens or []  # 保存特殊token列表

        # ========== 优化2：构建反向词表 ==========
        # 【原理】
        # 原始词表是 ID → 字节 的映射
        # 编码时需要频繁进行 字节 → ID 的查找
        # 所以预先构建一个反向字典，避免每次都遍历查找
        #
        # 【示例】
        # vocab = {0: b'a', 1: b'b', 256: b'th'}
        # reverted_vocab = {b'a': 0, b'b': 1, b'th': 256}
        # reverted_vocab[bytes] = id，可以快速从字节表示查找token ID
        # 【PyTorch替换说明】
        # 如果词表很大，可以考虑用torch.index_put或torch.scatter构建反向映射
        # 但Python字典在这种场景下已经非常高效
        self.reverted_vocab = {v: k for k, v in self.vocab.items()}

        # ========== 优化3：引入缓存机制 （加速BPE的关键）==========
        # 【原理】这是BPE加速的关键！
        # 很多文本中会出现相同的词，比如 "the" 可能出现很多次
        # 如果每次都重新计算BPE，会浪费大量时间
        # 我们把已经计算过的结果缓存起来，下次直接使用
        #
        # 【示例】
        # 第一次编码 "the"：
        #   - 计算BPE：[b't', b'h', b'e'] → [b'the'] → ID [257]
        #   - 缓存：self.cache[b'the'] = [257]
        # 第二次编码 "the"：
        #   - 直接从缓存读取：[257]
        #   cache[bytes] = list[int]，缓存已编码的token序列
        # 【PyTorch替换说明】
        # 如果需要在GPU上运行，可以用torch的tensor缓存
        # 但对于CPU场景，Python字典已经足够快
        self.cache: Dict[bytes, List[int]] = {}

        # ========== 处理特殊token ==========
        # 特殊token需要特殊处理：
        # 1. 如果特殊token已经在词表中，记录它的ID
        # 2. 如果不在词表中，给它分配一个新的ID
        self.special_token_to_id: Dict[str, int] = {}
        if self.special_tokens:
            for tok in self.special_tokens:
                tok_bytes = tok.encode("utf-8")  # 把特殊token转成字节
                # 在词表中查找这个特殊token
                found_ids = [i for i, b in self.vocab.items() if b == tok_bytes]
                if found_ids:
                    # 如果找到了，记录它的ID
                    self.special_token_to_id[tok] = found_ids[0]
                else:
                    # 如果没找到，给它分配一个新的ID
                    # 新ID = 当前词表最大ID + 1
                    new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                    self.vocab[new_id] = tok_bytes  # 添加到词表
                    self.special_token_to_id[tok] = new_id  # 记录ID
                    self.reverted_vocab[tok_bytes] = new_id  # 添加到反向词表

        # ========== GPT-2的预处理分词正则表达式 ==========
        # 【什么是预处理分词？】
        # 在BPE编码之前，先用正则表达式把文本切分成"词片段"
        # 这样可以：
        # 1. 把单词和标点分开（避免 "hello," 被当作一个整体）
        # 2. 处理缩写形式（如 "I'll", "don't"）
        # 3. 保留空格信息（GPT-2会在词前加空格，如 " hello"）
        #
        # 【正则表达式详解】
        # 这个正则表达式会把文本切分成以下几类：
        #
        # 1. '(?:[sdmt]|ll|ve|re) - 匹配缩写后缀
        #    例如：'s (is/has), 'd (would), 'm (am), 't (not), 'll (will), 've (have), 're (are)
        #    匹配：I'm 中的 'm，don't 中的 't，I'll 中的 'll
        #
        # 2. ' ?\p{L}+ - 匹配单词（可选前导空格 + 字母）
        #    \p{L} 匹配任何Unicode字母（包括中文、日文等）
        #    例如："hello", " world", "你好"
        #
        # 3. ' ?\p{N}+ - 匹配数字（可选前导空格 + 数字）
        #    \p{N} 匹配任何Unicode数字
        #    例如："123", " 456"
        #
        # 4. ' ?[^\s\p{L}\p{N}]+ - 匹配标点符号（可选前导空格 + 非字母非数字非空白）
        #    例如：",", "!", ".", "..."
        #
        # 5. '\s+(?!\S) - 匹配末尾空白（后面不跟非空白字符的空白）
        #    (?!\S) 是负向先行断言，确保空白后面没有非空白字符
        #
        # 6. '\s+ - 匹配其他空白
        #
        # 【示例】
        # 输入："Hello, I'm learning NLP!"
        # 输出：["Hello", ",", " I", "'m", " learning", " NLP", "!"]
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # ========== 构建特殊token的正则表达式 ==========
        # 如果有特殊token，需要构建一个正则表达式来识别它们
        # 这样在编码时可以正确处理特殊token，不会被BPE拆分
        if self.special_tokens:
            # 把所有特殊token用 | 连接起来，用 () 分组
            # 例如：'( <|endoftext|>|<|pad|>)'
            specials_pat = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            self.specials_regex = re.compile(specials_pat)
        else:
            self.specials_regex = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] | None = None):
        """
        从文件加载BPE分词器（类方法）

        【使用场景】
        通常我们不会手动创建词表和合并规则，而是从预训练的文件中加载。
        GPT-2的预训练词表可以从OpenAI获取。


       深度学习小知识：
        - 通常BPE分词器需要两个文件：
          1. vocab.json：词表文件，包含token ID到字节表示的映射
          2. merges.txt：合并规则文件，包含BPE学习到的合并顺序

        参数说明：
        ----------
        vocab_filepath : str
            词表文件的路径（JSON格式）
            文件内容示例：{"0": "a", "1": "b", "256": "th", "257": "the"}

        merges_filepath : str
            合并规则文件的路径（JSON格式）
            文件内容示例：[["t", "h"], ["th", "e"]]

        special_tokens : List[str] | None
            特殊token列表（可选）

        返回：
        ------
        Tokenizer 初始化好的分词器实例

        【PyTorch替换说明】
        文件加载是I/O密集型操作，PyTorch无法优化这部分。
        如果词表很大，可以考虑用mmap（内存映射）来加速加载。
        """
        # 读取词表文件
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            vocab_json = json.load(f)
        # JSON中的key是字符串，需要转成整数
        # JSON中的value是字符串，需要转成字节（用latin1编码，因为每个字符对应一个字节）
        # 将JSON数据转换为 {int: bytes} 格式
        vocab = {int(k): v.encode("latin1") for k, v in vocab_json.items()}

        # 读取合并规则文件
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_json = json.load(f)
        # 同样需要把字符串转成字节 将合并规则转换为 [(bytes, bytes)] 格式
        merges = [(p[0].encode("latin1"), p[1].encode("latin1")) for p in merges_json]

        # 调用普通的构造函数
        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: bytes) -> List[int]:
        """
        对单个预分词片段（pre-token）进行BPE编码（带缓存优化）

        【BPE编码算法详解】

        输入：字节序列，例如 b'hello'
        输出：token ID列表，例如 [104, 101, 271, 111]

        算法步骤：
        ---------
        第1步：初始化
        - 把字节序列拆成单字节列表
        - 例如：b'hello' → [b'h', b'e', b'l', b'l', b'o']

        第2步：查找最佳合并对
        - 遍历所有相邻的字节对
        - 在合并规则中查找每个字节对的优先级（排名）
        - 选择排名最高（排名数值最小）的字节对
        - 例如：假设 (b'l', b'l') 的排名是 50，其他字节对排名更大
        - 那么选择 (b'l', b'l') 作为最佳合并对

        第3步：执行合并
        - 把所有匹配的字节对合并成一个新的字节序列
        - 例如：[b'h', b'e', b'l', b'l', b'o'] → [b'h', b'e', b'll', b'o']

        第4步：重复第2-3步
        - 继续查找最佳合并对并合并
        - 直到没有可合并的字节对为止

        第5步：查词表得到ID
        - 把最终的字节列表转换成token ID
        - 例如：[b'h', b'e', b'll', b'o'] → [104, 101, 271, 111]

        【缓存优化】
        如果这个字节序列之前处理过，直接从缓存返回结果，避免重复计算。




        参数说明：
            token_bytes (bytes): 待编码的字节序列


        返回：
            List[int]: token ID列表

        【PyTorch替换说明】
        这个方法的核心是字符串匹配和合并，不太适合用PyTorch优化。
        但如果要处理大批量数据，可以考虑：
        1. 用torch.jit.script编译这个方法（可能有小幅提升）
        2. 用Cython或Rust重写核心循环（更显著的提升）
        """
        # ========== 检查缓存 ==========
        # 如果这个字节序列之前处理过，直接返回缓存的结果
        # 这是BPE加速的关键优化之一！
        if token_bytes in self.cache:
            return self.cache[token_bytes]

        # ========== 初始化：把字节序列拆成单字节列表 ==========
        # bytes([b]) 把单个整数转成单字节的bytes对象
        # 例如：b'hello' → [b'h', b'e', b'l', b'l', b'o']
        word = [bytes([b]) for b in token_bytes]

        # ========== 主循环：不断合并直到无法合并 ==========
        while len(word) > 1:  # 至少要有2个元素才能合并
            # 找出所有相邻的字节对
            # 例如：[b'h', b'e', b'l', b'l', b'o'] 的字节对是：
            # [(b'h', b'e'), (b'e', b'l'), (b'l', b'l'), (b'l', b'o')]
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]

            # 找到优先级最高的字节对（排名ranks最小）
            # 【为什么是ranks最小？】
            # Python的heapq只支持小顶堆（最小的元素在堆顶）。
            # 我们要找最大频率，所以用负数来实现大顶堆。
            # 例如：频率4变成-4，频率2变成-2。
            # 堆顶是最小的负数（即最大的频率）。
            # self.ranks.get(p, float('inf')) 的含义：
            # - 如果字节对p在合并规则中，返回它的排名
            # - 如果不在，返回无穷大（表示不能合并）
            # min() 找到排名最小的字节对
            best_pair = min(pairs, key=lambda p: self.ranks.get(p, float('inf')))

            # 如果最佳字节对不在合并规则中，说明没有可合并的了，退出循环
            if best_pair not in self.ranks:
                break

            # ========== 执行合并 ==========
            # 把所有匹配 best_pair 的相邻字节对合并
            new_word = []
            i = 0
            p1, p2 = best_pair  # 解包字节对，例如 (b'l', b'l')

            while i < len(word):
                # 检查当前位置是否匹配要合并的字节对
                # 如果当前位置是p1，下一个位置是p2，则合并它们
                if i < len(word) - 1 and word[i] == p1 and word[i + 1] == p2:
                    # 匹配成功，合并这两个字节
                    # p1 + p2 把两个bytes对象拼接成一个新的bytes
                    # 例如：b'l' + b'l' = b'll'
                    new_word.append(p1 + p2)
                    i += 2  # 跳过已合并的两个字节
                else:
                    # 不匹配，保留原字节
                    new_word.append(word[i])
                    i += 1

            # 更新word，继续下一轮合并
            word = new_word

        # ========== 查词表得到token ID ==========
        # 把最终的字节列表转换成token ID列表
        ids = [self.reverted_vocab[tok] for tok in word]

        # ========== 存入缓存 ==========
        # 把结果缓存起来，下次遇到相同的字节序列直接使用
        self.cache[token_bytes] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        """
        将文本编码为token ID序列（对外接口）

        【编码流程】

        输入："Hello, world!"

        第1步：处理特殊token
        - 如果有特殊token（如  ），先用正则表达式分割文本
        - 例如："Hello,  world!" → ["Hello, ", "  ", " world!"]

        第2步：对每个片段进行处理
        - 如果片段是特殊token，直接添加它的ID
        - 如果不是，用正则表达式切分成词片段
        - 对每个词片段进行BPE编码

        第3步：返回所有token ID的列表

        【示例】
        输入："Hello, world!"
        特殊token：[]

        第1步：没有特殊token，segments = ["Hello, world!"]

        第2步：用正则表达式切分
        - "Hello, world!" → ["Hello", ",", " world", "!"]

        第3步：对每个词片段进行BPE编码
        - "Hello" → [15496, 2159]  (假设)
        - "," → [11]
        - " world" → [995]
        - "!" → [0]

        输出：[15496, 2159, 11, 995, 0]


        参数说明：
            text (str): 待编码的文本


        返回：
            List[int]: token ID列表

        【PyTorch替换说明】
        如果需要批量编码大量文本，可以考虑：
        1. 用torch.utils.data.DataLoader并行处理
        2. 用torch.jit.script编译这个方法
        但对于单个文本编码，Python实现已经足够快。
        """
        token_ids = []

        # ========== 第1步：处理特殊token ==========
        # 如果有特殊token，先用正则表达式分割文本
        # 这样可以确保特殊token不会被BPE拆分
        if self.specials_regex:
            segments = self.specials_regex.split(text)
        else:
            segments = [text]

        # ========== 第2步：对每个片段进行处理 ==========
        # 对每个片段进行编码
        for seg in segments:
            # 跳过空片段
            if not seg: continue

            # 如果这个片段是特殊token，直接添加它的ID
            if seg in self.special_token_to_id:
                token_ids.append(self.special_token_to_id[seg])
            else:
                # 否则，用正则表达式切分成词片段
                # finditer 使用正则表达式匹配pre-token并返回所有匹配的迭代器
                for m in self.pat.finditer(seg):
                    # m.group(0) 是匹配到的文本
                    # 转成UTF-8字节序列
                    pre_token_bytes = m.group(0).encode("utf-8")
                    # 调用_bpe方法对pre-token进行BPE编码
                    # extend把返回的ID列表添加到结果中
                    token_ids.extend(self._bpe(pre_token_bytes))

        return token_ids

    def decode(self, ids: List[int]) -> str:
        """
        将token ID序列解码为文本（对外接口）

        【解码流程】

        输入：[15496, 2159, 11, 995, 0]

        第1步：查词表得到字节序列
        - 根据每个ID在词表中查找对应的字节序列
        - 例如：15496 → b'Hello', 2159 → b',', 11 → b' world', 0 → b'!'

        第2步：拼接所有字节
        - b'Hello' + b',' + b' world' + b'!' = b'Hello, world!'

        第3步：用UTF-8解码成字符串
        - b'Hello, world!' → "Hello, world!"

        【错误处理】
        - errors="replace" 表示如果遇到无法解码的字节，用替换字符（�）代替
        - 这样可以避免解码失败




        参数说明：
            ids (List[int]): token ID列表


        返回：
            str: 解码后的文本

        【PyTorch替换说明】
        这个方法非常简单，主要是字典查找和字节拼接。
        如果需要批量解码，可以考虑：
        1. 用torch.gather批量查词表
        2. 用torch.cat拼接字节序列
        但对于单个序列解码，Python实现已经足够快。
        """
        # ========== 第1步：查词表得到字节序列 ==========
        # 根据每个ID在词表中查找对应的字节序列
        # 例如：ids = [15496, 2159, 11]
        # vocab[15496] = b'Hello'
        # vocab[2159] = b','
        # vocab[11] = b' world'
        # byte_stream = b'Hello, world'
        byte_stream = b"".join(self.vocab[idx] for idx in ids)

        # ========== 第2步：用UTF-8解码成字符串 ==========
        # errors="replace" 表示如果遇到无法解码的字节，用替换字符（�）代替
        # 这样可以避免解码失败
        return byte_stream.decode('utf-8', errors="replace")


'''
tiktoken 的优化实现
1. O(1)查询优化
✅ tiktoken 实现了：内部使用 Rust 实现的高性能数据结构，合并规则查询是 O(1) 的


2. 预构建反向词汇表
✅ tiktoken 实现了：在初始化时构建完整的索引结构


3. 缓存机制
✅ tiktoken 实现了：内置 LRU 缓存，自动缓存编码结果


强烈推荐替换，因为：


✅ tiktoken 实现了所有三个优化，性能更好
✅ 官方维护，稳定性更高
✅ Rust 实现，速度快 10-100 倍
✅ 内存效率更高
✅ 支持最新的 GPT 模型分词规则
'''
# import tiktoken
#
# # tiktoken 的性能优势：
# # 1. Rust 实现，比纯 Python 快 10-100 倍
# # 2. 内存效率更高
# # 3. 支持多线程编码
# # 4. 官方维护，稳定性更好
# ```
#
# ## 替换示例
#
# ```python
# import tiktoken
#
# # 初始化 tiktoken 分词器（相当于你的 Tokenizer 类）
# tokenizer = tiktoken.get_encoding("gpt2")
#
# # 编码（相当于你的 encode 方法）
# text = "Hello, world!"
# token_ids = tokenizer.encode(text)
#
# # 解码（相当于你的 decode 方法）
# decoded_text = tokenizer.decode(token_ids)
#
# # 特殊token处理
# # tiktoken 支持特殊token，但API略有不同
# token_ids_with_special = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# ```
#
# ## 性能对比
#
# ```python
# import time
# import tiktoken
#
# # 你的实现 vs tiktoken
# text = "Hello, world! " * 1000
#
# # 你的实现（估算）
# # 编码时间：~100-500ms（取决于缓存命中率）
#
# # tiktoken 实现
# tokenizer = tiktoken.get_encoding("gpt2")
# start = time.time()
# ids = tokenizer.encode(text)
# tiktoken_time = time.time() - start
# # 编码时间：~1-5ms（快 20-100 倍）
# ```
#
# ## API 差异需要注意
#
# 1. ** 特殊token处理 **：
# ```python
# # 你的实现
# tokenizer = Tokenizer(vocab, merges, special_tokens=["<eos>", "<pad>"])
#
# # tiktoken
# tokenizer = tiktoken.get_encoding("cl100k_base")
# ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# ```
#
# 2. ** 缓存控制 **：
# ```python
# # 你的实现：手动管理 cache 字典
# # tiktoken：自动缓存，无需手动管理
# ```
#
# 3. ** 配置加载 **：
# ```python
# # 你的实现：从文件加载
# tokenizer = Tokenizer.from_files("vocab.json", "merges.txt")
#
# # tiktoken：使用预设编码
# tokenizer = tiktoken.get_encoding("gpt2")  # 或 "cl100k_base", "p50k_base" 等
# ```
#
# ## 推荐替换方案
#
# ```python
# import tiktoken
# from typing import List, Optional
#
#
# class TikTokenizerWrapper:
#     """tiktoken 的包装器，保持与你原有 API 兼容"""
#
#     def __init__(self, encoding_name: str = "gpt2", special_tokens: Optional[List[str]] = None):
#         self.tokenizer = tiktoken.get_encoding(encoding_name)
#         self.special_tokens = special_tokens or []
#
#     def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
#         if allowed_special is None and self.special_tokens:
#             allowed_special = set(self.special_tokens)
#         return self.tokenizer.encode(text, allowed_special=allowed_special)
#
#     def decode(self, ids: List[int]) -> str:
#         return self.tokenizer.decode(ids)
#
#
# # 使用方式（与你的 API 兼容）
# tokenizer = TikTokenizerWrapper("gpt2", special_tokens=["<|endoftext|>"])
# token_ids = tokenizer.encode("Hello, world!")
# text = tokenizer.decode(token_ids)
