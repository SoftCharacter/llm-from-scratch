"""
将原始文本数据编码为模型训练需要的uint16二进制文件。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tokenizer_optimized import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_DIR = PROJECT_ROOT / "trained_tokenizer"
DEFAULT_DATA_DIR = PROJECT_ROOT / "tokenized_data"
DEFAULT_VOCAB_PATH = DEFAULT_TOKENIZER_DIR / "vocab_of_your_tokenizer.json"
DEFAULT_MERGES_PATH = DEFAULT_TOKENIZER_DIR / "merges_of_your_tokenizer.json"
DEFAULT_TRAIN_BIN = "your_train_data.bin"
DEFAULT_VAL_BIN = "your_val_data.bin"
UINT16_MAX = np.iinfo(np.uint16).max


def encode_text_file(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> int:
    """
    读取txt文本，编码为token ID，并保存为uint16二进制文件。
    """
    text = input_path.read_text(encoding="utf-8", errors="replace")
    token_ids = tokenizer.encode(text)

    if token_ids and max(token_ids) > UINT16_MAX:
        raise ValueError(
            f"token id超过uint16上限{UINT16_MAX}，请改用更大的dtype或降低vocab_size。"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(token_ids, dtype=np.uint16).tofile(output_path)
    return len(token_ids)


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Encode train/val txt files into uint16 bin files.")
    parser.add_argument("--train_txt", required=True, help="Path to raw training txt file.")
    parser.add_argument("--val_txt", required=True, help="Path to raw validation txt file.")
    parser.add_argument("--tokenizer_vocab", default=str(DEFAULT_VOCAB_PATH), help="Path to tokenizer vocab JSON.")
    parser.add_argument("--tokenizer_merges", default=str(DEFAULT_MERGES_PATH), help="Path to tokenizer merges JSON.")
    parser.add_argument("--output_dir", default=str(DEFAULT_DATA_DIR), help="Directory to save output bin files.")
    parser.add_argument("--train_bin", default=DEFAULT_TRAIN_BIN, help="Output training bin filename.")
    parser.add_argument("--val_bin", default=DEFAULT_VAL_BIN, help="Output validation bin filename.")
    parser.add_argument(
        "--special_token",
        action="append",
        default=None,
        help="Special token used by the tokenizer. Repeat this argument to add more tokens.",
    )
    return parser.parse_args()


def main() -> None:
    """
    将训练集和验证集txt分别编码为bin文件。
    """
    args = parse_args()
    special_tokens = args.special_token or ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(args.tokenizer_vocab, args.tokenizer_merges, special_tokens=special_tokens)

    output_dir = Path(args.output_dir)
    train_output = output_dir / args.train_bin
    val_output = output_dir / args.val_bin

    train_count = encode_text_file(tokenizer, Path(args.train_txt), train_output)
    val_count = encode_text_file(tokenizer, Path(args.val_txt), val_output)

    print(f"Saved train data: {train_output} ({train_count} tokens)")
    print(f"Saved val data: {val_output} ({val_count} tokens)")


if __name__ == "__main__":
    main()
