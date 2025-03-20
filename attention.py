import torch
import torch.nn as nn

# input: (batch_size, seq_len, d_model)

# class Attention(nn.Module):

# I love coding

# I [1, 2,3] love [4, 5, 6] coding [7, 8, 9]


#why bias = False in Q, K, V?

# In attention mechanisms, particularly in transformer architectures, it's common to set bias=False in the linear transformations for query, key, and value projections for several reasons:

# Theoretical Motivation:
# The attention mechanism is fundamentally about computing similarities between vectors
# Adding a bias term doesn't meaningfully contribute to this similarity computation
# The core attention operation (Q·K^T) is about measuring relative relationships between elements, not absolute values
# Mathematical Perspective:
# In the scaled dot-product attention formula: Attention(Q,K,V) = softmax(QK^T/√d)V
# The softmax operation is invariant to constant shifts (which is what a bias would introduce)
# Any bias added to the query or key would get "washed out" by the softmax normalization
# Empirical Evidence:
# Research has shown that removing bias terms from Q,K,V projections doesn't hurt performance
# It reduces the number of parameters without significant impact on model quality
# Many popular transformer implementations (like the original "Attention Is All You Need" paper) use bias-free projections
# Looking at your current code, you're using nn.Parameter directly. If you want to switch to using nn.Linear, here's how you could modify it:



# Simple attention
class Attention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W_q = nn.Linear(dim_in, dim_out, bias=False)
        self.W_k = nn.Linear(dim_in, dim_out, bias=False)
        self.W_v = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x):
   

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        attention_scores = q @ k.T
        attention_weights = torch.softmax(attention_scores / torch.sqrt(torch.tensor(k.shape[-1])), dim=-1)
        context_vector = attention_weights @ v

        return context_vector


#Masked attention

class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W_q = nn.Linear(dim_in, dim_out, bias=False)
        self.W_k = nn.Linear(dim_in, dim_out, bias=False)
        self.W_v = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        
    
    


X = torch.randn(3, 4)
dim_in = 4
dim_out = 10

attention = Attention(dim_in, dim_out)
print(attention(X))