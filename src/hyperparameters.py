from dataclasses import dataclass

from torch import cuda


@dataclass
class Hyperparameters:
    device = "cuda" if cuda.is_available() else "cpu"

    # batch_size = 32  # how many independent sequences will we process in parallel?
    # block_size = 8  # what is the max context length for predictions?
    #
    # num_layers = 4
    # num_embds = 32
    # num_heads = 4
    # dropout = 0.2
    #
    # max_iters = 5000
    # eval_iters = 200
    # eval_interval = 300
    # learning_rate = 1e-3
    #
    # max_new_tokens = 500

    """
    Scaled up
    """

    batch_size = 64  # how many independent sequences will we process in parallel?
    block_size = 256  # what is the max context length for predictions?

    num_embds = 384
    num_heads = 6
    num_layers = 6
    dropout = 0.2

    max_iters = 5000
    eval_iters = 500
    eval_interval = 500
    learning_rate = 3e-4

    max_new_tokens = 500
