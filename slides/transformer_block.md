# Transformer

<br>
<br>

The building block of modern AI models.

They consist of one Attention module*, and two linear projections (feed-forward network).

```python
Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

FFN(x) = activation(x @ W1 + b1) @ W2 + b2
```

<br>

<small>* MHA, GQA, MLA, whichever attention variant a given model actually uses.</small>
