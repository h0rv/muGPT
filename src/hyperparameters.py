from dataclasses import dataclass

from torch import cuda


@dataclass
class Hyperparameters:
    device = "cuda" if cuda.is_available() else "cpu"

    batch_size = 32  # how many independent sequences will we process in parallel?
    block_size = 8  # what is the max context length for predictions?

    n_embd = 32
    head_size = n_embd
    num_heads = 4
    max_iters = 5000
    learning_rate = 1e-3
    eval_interval = 300
    eval_iters = 200

    max_new_tokens = 500
