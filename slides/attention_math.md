# It's just matrices

Confused by "query", "key", "value"? Drop the vocabulary for a second: they
are three matrices, made by multiplying the input by three learned
weight matrices.

```python
Q = X @ W_q
K = X @ W_k
V = X @ W_v

Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V
```

`Q @ K.T` compares every query against every key. Softmax turns
those scores into a row of percentages. Multiplying by `V` blends the values
by those percentages. Three matrices, two matrix multiplications, one
softmax.
