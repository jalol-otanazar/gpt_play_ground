import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper parameters
batch_size = 64
block_size = 256
lr = 3e-4
max_iters = 5000
eval_iters = 200
eval_interval = 500
n_emb = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('input.txt', 'r', encoding='utf8') as f:
    text = f.read()
# tokenization
chars = sorted(set(list(text)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda num: ''.join(itos[i] for i in num)

# encode and split the text file
data =  torch.tensor(encode(text))

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# for reproduibility
torch.manual_seed(1234543)

@torch.no_grad()
def evaluate_loss():
    outs = {}
    m.eval()
    for mode in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(mode)
            logits, loss = m(x, y)
            losses[k] = loss.item()
        outs[mode] = losses.mean()
    m.train()
    return outs

# data loading, sampling
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, C)
        k = self.key(x) # (B, T, C)
        attr = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        attr = attr.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attr = F.softmax(attr, dim=-1)
        attr = self.dropout(attr)
        v = self.value(x) # (B, T, C)
        out = attr @ v # (B, T, C)
        return out
class MultiHead(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_emb, n_emb, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb, n_emb),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_emb, n_head):
        super().__init__()
        self.sa = MultiHead(n_head, n_emb//n_head)
        self.ffwd = FeedForward(n_emb)
        self.ln1 = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.pos_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(*[Block(n_emb, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size, bias=False)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            idx_crop = idx[:, -block_size:]
            logits, loss = self(idx_crop)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
model = BigramLanguageModel()
m = model.to(device)

#optimizer creation
optimizer = torch.optim.AdamW(m.parameters(), lr)

# the loop
for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = evaluate_loss()
        print(f"step:{iter} train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")
    # minibatch
    Xb, Yb = get_batch('train')
    
    # evaluation
    logits, loss = m(Xb, Yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, 1000)[0].tolist()))