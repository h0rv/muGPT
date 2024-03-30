import math
from torch.nn import functional as F
import torch.nn as nn
import torch
import csv


"""
Hyperparameters
"""

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
""""""

"""
kaggle datasets download -d eswarreddy12/family-guy-dialogues-with-various-lexicon-ratings
mkdir -p data
unzip family-guy-dialogues-with-various-lexicon-ratings.zip -d data
rm family-guy-dialogues-with-various-lexicon-ratings.zip
"""
file_path = "data/Family_Guy_Final_NRC_AFINN_BING.csv"

# Print headings
with open(file_path, "r") as f:
    csvf = csv.reader(f, delimiter=",", quotechar="'")
    headings = next(csvf)  # Read the first row (headings)
    print(", ".join(headings))

corpus = []

# Read dialogue and create the corpus
with open(file_path, "r") as f:
    csvf = csv.DictReader(f, delimiter=",", quotechar="'")
    for row in csvf:
        dialogue = row["Dialogue"].strip('"')  # Remove double quotes
        corpus.append(dialogue)

# Print the first 5 entries of the corpus
print("Corpus:")
for i in range(5):
    print(corpus[i])

print(f"Corpus line num: {len(corpus)}")

corpus_str = "\n".join(corpus)

print(f"Corpus str len: {len(corpus_str)}")

print(corpus_str[:1000])

# Unique chars in the corpus

chars = sorted(list(set(corpus_str)))
vocab_size = len(chars)
print("".join(chars))
print(f"Vocab size: {vocab_size}")

# TODO: would need to remove chars from corpus too
# chars = chars[:91] # remove special chars
# vocab_size = len(chars)
# print(''.join(chars))
# print(f"Vocab size: {vocab_size}")

# Create mapping from chars to ints

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return "".join([itos[i] for i in l])


print(encode("Hi there!"))
print(decode(encode("Hi there!")))


data = torch.tensor(encode(corpus_str), dtype=torch.long)

print(data.shape)
print(data.dtype)
print(data[:100])

# Split dataset into train and validation
n = int(0.9 * len(corpus_str))

train_data, val_data = data[:n], data[n:]

block_size = 8
train_data[: block_size + 1]

x = train_data[:block_size]
y = train_data[1 : block_size + 1]

for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"when input is {context} the target is {target}")

torch.manual_seed(1337)

batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the max context length for predictions?


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x, y


(
    xb,
    yb,
) = get_batch("train")

print("inputs:")
print(xb.shape)
print(xb)
print("targets:")
print(yb.shape)
print(yb)

print("\n-----\n")

for b in range(batch_size):  # batch dim
    for t in range(block_size):  # time dim
        context = xb[b, : t + 1]
        target = yb[b, 1]
        print(f"when input is {context.tolist()} the target is {target}")


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of ints
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # (B, C, T)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context

        for _ in range(max_new_tokens):
            # get predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # ==> (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print("Expected loss: " + str(-math.log(1 / vocab_size)))

print("\n-----\n")

idx = torch.zeros((1, 1), dtype=torch.long)

generated = decode(m.generate(idx, max_new_tokens=100)[0].tolist())

print(generated)

# Train model
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):
    # sample a batch of data
    xb, yb = get_batch("train")

    # eval the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

generated = decode(m.generate(idx, max_new_tokens=500)[0].tolist())
print(generated)
