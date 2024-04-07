import torch

from bigram_language_model import BigramLanguageModel
from corpus import get_corpus
from hyperparameters import Hyperparameters as HP

torch.manual_seed(1337)


corpus = get_corpus()

"""
Encoding/Decoding
"""
chars = sorted(list(set(corpus)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


"""
Split dataset into train and validation
"""
data = torch.tensor(encode(corpus), dtype=torch.long)
n = int(0.9 * len(corpus))
train_data, val_data = data[:n], data[n:]


"""
Data loading
"""
x = train_data[: HP.block_size]
y = train_data[1 : HP.block_size + 1]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - HP.block_size, (HP.batch_size,))

    x = torch.stack([data[i : i + HP.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + HP.block_size + 1] for i in ix])

    x, y = x.to(HP.device), y.to(HP.device)

    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}

    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(HP.eval_iters)

        for k in range(HP.eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    model.train()

    return out


model = BigramLanguageModel(
    vocab_size,
    HP.num_embds,
    HP.block_size,
    HP.num_heads,
    HP.num_heads,
    HP.dropout,
    HP.device,
)
m = model.to(HP.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=HP.learning_rate)

"""
Training loop
"""
for iter in range(HP.max_iters):
    if iter % HP.eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # eval the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


"""
Generate from model
"""
context = torch.zeros((1, 1), dtype=torch.long, device=HP.device)
print(decode(m.generate(context, max_new_tokens=HP.max_new_tokens)[0].tolist()))
