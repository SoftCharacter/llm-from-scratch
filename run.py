"""
使用Python启动模型训练，功能和参数与run.sh保持一致。
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


TRAIN_DATA = "tokenized_data/your_train_data.bin"
VAL_DATA = "tokenized_data/your_val_data.bin"
VOCAB = "trained_tokenizer/vocab_of_your_tokenizer.json"
MERGES = "trained_tokenizer/merges_of_your_tokenizer.json"
OUT_ROOT = "train_logs"


def main() -> None:
    """
    创建本次训练日志目录，并以后台进程启动训练脚本。
    """
    work_dir = Path.cwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(OUT_ROOT) / f"run_{timestamp}"
    log_file = out_dir / "train.log"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=====================================================")
    print(f"Work Dir: {work_dir}")
    print(f"Output Dir: {out_dir}")
    print(f"Log File: {log_file}")
    print("=====================================================")

    command = [
        sys.executable,
        "-u",
        "main/run_train_model.py",
        "--train_data", TRAIN_DATA,
        "--val_data", VAL_DATA,
        "--tokenizer_vocab", VOCAB,
        "--tokenizer_merges", MERGES,
        "--out_dir", str(out_dir),
        "--batch_size", "64",
        "--max_iters", "4200",
        "--eval_interval", "100",
        "--eval_iters", "20",
        "--log_interval", "10",
        "--vocab_size", "10000",
        "--context_length", "256",
        "--n_head", "16",
        "--theta", "10000",
        "--n_layers", "4",
        "--d_model", "512",
        "--d_ff", "1344",
        "--residual_type", "standard",
        "--attn_residual_window", "0",
        "--weight_decay", "1e-1",
        "--max_norm", "1.0",
        "--max_lr", "6e-4",
        "--min_lr", "6e-5",
        "--warmup_iters", "200",
        "--lr_decay_iters", "3600",
    ]

    log_handle = open(log_file, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=work_dir,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    print(f"Training started with PID: {process.pid}")
    print(f"To monitor the log: tail -f {log_file}")
    print("To try attention residuals, change --residual_type to attention.")


if __name__ == "__main__":
    main()
