import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(123)

# ============================================================
# 1) Multi-Head Causal Self-Attention (GPT-style)
#    - "causal" means: token i cannot look at future tokens j>i
# ============================================================
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, max_seq_len=128):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # One projection per Q/K/V (each outputs embed_dim, then we split into heads)
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)

        # Output projection after concatenating heads
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Precompute a causal mask (lower triangular)
        # shape: (max_seq_len, max_seq_len)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask)

    def forward(self, x, debug=False):
        """
        x: (batch, seq_len, embed_dim)
        returns:
          out : (batch, seq_len, embed_dim)
          attn: (batch, heads, seq_len, seq_len)
        """
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        # 1) Q,K,V projections (still (B,T,C))
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # 2) Split into heads:
        # (B,T,C) -> (B,T,H,D) -> (B,H,T,D)
        Q = Q.view(B, T, H, D).transpose(1, 2)
        K = K.view(B, T, H, D).transpose(1, 2)
        V = V.view(B, T, H, D).transpose(1, 2)

        # 3) Attention scores:
        # (B,H,T,D) @ (B,H,D,T) -> (B,H,T,T)
        scores = (Q @ K.transpose(-2, -1)) * self.scale

        # 4) Causal mask: block future positions (j > i)
        # allowed positions = 1, blocked = 0 -> set blocked to -inf so softmax -> 0
        m = self.causal_mask[:T, :T]  # (T,T)
        scores = scores.masked_fill(m == 0, float("-inf"))

        # 5) Softmax -> attention weights (each row sums to 1)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 6) Weighted sum of V:
        # (B,H,T,T) @ (B,H,T,D) -> (B,H,T,D)
        ctx = attn @ V

        # 7) Concat heads: (B,H,T,D) -> (B,T,C)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, C)

        # 8) Output projection
        out = self.Wo(ctx)

        if debug:
            print("\n[ATTENTION]")
            print(f"  x:      {tuple(x.shape)}  (B,T,C)")
            print(f"  Q,K,V:  {tuple(Q.shape)}  (B,H,T,D)")
            print(f"  scores: {tuple(scores.shape)}  (B,H,T,T)")
            print(f"  attn:   {tuple(attn.shape)}  (B,H,T,T)")
            print(f"  out:    {tuple(out.shape)}  (B,T,C)")
            print("  head0 attention (first 4x4):")
            block = attn[0, 0, :4, :4].detach()
            for i in range(block.size(0)):
                print("   ", "  ".join(f"{v.item():.3f}" for v in block[i]))

        return out, attn


# ============================================================
# 2) FFN / MLP (position-wise)
#    expand -> GELU -> shrink
# ============================================================
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)   # expand
        self.fc2 = nn.Linear(hidden_dim, embed_dim)   # shrink
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, debug=False):
        h = self.fc1(x)       # (B,T,hidden_dim)
        h = F.gelu(h)         # non-linearity
        h = self.dropout(h)
        out = self.fc2(h)     # (B,T,embed_dim)
        out = self.dropout(out)

        if debug:
            print("\n[FFN]")
            print(f"  in:      {tuple(x.shape)}  (B,T,embed_dim)")
            print(f"  expanded:{tuple(h.shape)}  (B,T,hidden_dim)")
            print(f"  out:     {tuple(out.shape)}  (B,T,embed_dim)")

        return out


# ============================================================
# 3) Transformer Block (GPT uses Pre-LN typically)
#    x = x + Attention(LN(x))
#    x = x + FFN(LN(x))
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.0, max_seq_len=128):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalMultiHeadAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, hidden_dim, dropout)

    def forward(self, x, debug=False):
        # Attention + residual
        attn_out, attn_w = self.attn(self.ln1(x), debug=debug)
        x = x + attn_out

        # FFN + residual
        ffn_out = self.ffn(self.ln2(x), debug=debug)
        x = x + ffn_out

        return x, attn_w


# ============================================================
# 4) GPT-style model
#    - token embedding + positional embedding
#    - stack of blocks
#    - final LN + linear head to vocab logits
# ============================================================
class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, hidden_dim, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids, debug=False):
        """
        input_ids: (B,T)
        logits: (B,T,V)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, "Sequence too long for configured max_seq_len"

        # 1) Embeddings
        positions = torch.arange(T, device=input_ids.device)          # (T,)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)         # (B,T,C)
        x = self.drop(x)

        if debug:
            print("=" * 70)
            print("GPT FORWARD")
            print("=" * 70)
            print(f"[INPUT]  input_ids shape: {tuple(input_ids.shape)}  (B,T)")
            print(f"[EMBED]  x shape:        {tuple(x.shape)}  (B,T,embed_dim)")
            print(f"         vocab_size={self.vocab_size}, embed_dim={self.embed_dim}")
            print("-" * 70)

        # 2) Transformer blocks
        last_attn = None
        for i, blk in enumerate(self.blocks):
            # debug only first block to avoid spam
            x, last_attn = blk(x, debug=(debug and i == 0))
            if debug:
                print(f"[BLOCK {i}] x shape after block: {tuple(x.shape)}")

        # 3) Final norm + logits
        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)

        if debug:
            print("-" * 70)
            print(f"[FINAL]  logits shape: {tuple(logits.shape)}  (B,T,vocab_size)")

        return logits, last_attn


# ============================================================
# 5) DEMO: predict next word (IMPORTANT: model is UNTRAINED -> random-ish)
# ============================================================
# Tiny vocab (so we can print words easily)
vocab = {
    "<BOS>": 0,
    "i": 1,
    "will": 2,
    "write": 3,
    "a": 4,
    "feedback": 5,
    ".": 6
}
inv_vocab = {v: k for k, v in vocab.items()}

# Build a small GPT
model = GPT(
    vocab_size=len(vocab),
    embed_dim=32,
    num_heads=4,
    num_layers=2,
    max_seq_len=32,
    hidden_dim=128,  # typical 4x (32->128)
    dropout=0.0
)

# Input context: "<BOS> i will write a"
context_words = ["<BOS>", "i", "will", "write", "a"]
context_ids = torch.tensor([[vocab[w] for w in context_words]])  # (B=1,T=5)

print("\nContext:", " ".join(context_words))
print("Context IDs:", context_ids.tolist())

# Forward pass
logits, attn = model(context_ids, debug=True)

# We want "next word" after the last token in the context:
# take logits from last position (T-1)
last_logits = logits[0, -1]               # (V,)
probs = F.softmax(last_logits, dim=-1)    # (V,)

# Show top-k next-token predictions
topk = 5
values, indices = torch.topk(probs, k=min(topk, probs.numel()))

print("\n" + "=" * 70)
print("NEXT TOKEN PREDICTION (from last position)")
print("=" * 70)
for p, idx in zip(values, indices):
    print(f"  {inv_vocab[idx.item()]:<10}  prob={p.item():.3f}")

print("\nNOTE: This model is untrained, so predictions are not meaningful yet.")
print("After training on text, it learns realistic next-token probabilities.")



def gpt_generate(self, input_ids, max_new_tokens=3, temperature=1.0, top_k=None):
    """
    Autoregressive generation:
      - run model
      - take last position logits
      - sample next token
      - append and repeat
    """
    self.eval()
    generated = input_ids.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            # If sequence is too long, keep only last max_seq_len tokens
            current = generated[:, -self.max_seq_len:]

            logits, _ = self(current)                  # (B,T,V)
            last_logits = logits[:, -1, :] / temperature  # (B,V)

            # Optional top-k filtering (reduces randomness)
            if top_k is not None:
                vals, idxs = torch.topk(last_logits, k=top_k, dim=-1)
                filtered = torch.full_like(last_logits, float("-inf"))
                filtered.scatter_(1, idxs, vals)
                last_logits = filtered

            probs = F.softmax(last_logits, dim=-1)     # (B,V)
            next_token = torch.multinomial(probs, num_samples=1)  # (B,1)

            generated = torch.cat([generated, next_token], dim=1)

    return generated

# Monkey-patch onto your GPT class (quick way without editing the class definition)
GPT.generate = gpt_generate


# ------------------------------------------------------------
# Demo: generate 3 tokens after "i will write a"
# ------------------------------------------------------------
context_words = ["<BOS>", "i", "will", "write", "a"]
context_ids = torch.tensor([[vocab[w] for w in context_words]])

gen_ids = model.generate(context_ids, max_new_tokens=3, temperature=1.0, top_k=None)

# Decode
gen_words = [inv_vocab[t] for t in gen_ids[0].tolist()]
print("\nPROMPT:   ", " ".join(context_words))
print("GENERATED:", " ".join(gen_words))
print("NEW TOKENS ONLY:", " ".join(gen_words[len(context_words):]))