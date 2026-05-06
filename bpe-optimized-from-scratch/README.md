# bpe-optimized-from-scratch

An (extremely) optimized BPE tokenizer. Include codes for training and inference. From scratch.


## Performance Gain & Limitations

Achieved **9700x+** speedup in training and **3000x+** speedup in inference through algorithmic refinement. Algorithmically it reaches the near-optimal time complexity.

| Metric | Naive Implementation | Optimized Implementation | Improvement |
| :--- | :--- | :--- | :--- |
| **Training (it/s)** | 21.60 | **211,291.2** | **~9771x** |
| **Training Time (10GB)** | 19 hours | **7 seconds** | — |
| **Inference (Throughput)** | 830 B/s | **2,500,000 B/s** | **~3012x** |
| **Inference Time (10GB)** | 138 days | **0.5 hour** | — |

Nonetheless, the implementation has only been optimized at the algorithmic level using Python.
- A better implementation should consider hardware-level optimizations and use C++/Rust for further speedup.

## Project Structure

The structure of this repo is simple:
- The `bpe_slow/` stores my naive **from-scratch** implementation of BPE. It is more natural for new hands to understand the algorithm but not computationally efficient.
- The `bpe_optimized/` stores my optimized implementation of BPE. It is more complex but very efficient.
- A rough demostration of tokenizer (both training and inference) is in `play.ipynb`. 

## Optimization Strategy

1. Training Efficiency
    - **Weighted Space Reduction:** Aggregates identical tokens into frequency lists to minimize redundant data scanning.
    - **Inverted Indexing:** Maintains a mapping of byte-pairs to word locations for $O(1)$ local updates instead of global re-scanning.
    - **Lazy Max-Heap:** Utilizes a priority queue for merge candidates, deferring deletions to achieve amortized efficiency.

2. Inference Throughput
    - **Memoization (Zipf’s Law):** Caches BPE merge results for high-frequency subwords, bypassing the iterative merge process.
    - **Rank Dictionary:** Pre-computes merge rules into a hash map for $O(1)$ priority lookups.
    - **Vectorized Mapping:** Eliminates linear search overhead by pre-building reverse vocabulary maps.


## Quick Start

```python
from bpe_optimized.tokenizer import OptimizedTokenizer

# Usage details in play.ipynb
```


