# The knobs you already pay for

These are the parameters in every LLM API, and this is all they do:

- **temperature:** divide the scores before the softmax. Low = confident and
  repetitive. High = surprising, then incoherent.

```python
logits = logits / T
```
- **top-k:** only ever consider the *k* best tokens
- **top-p:** keep the smallest set of tokens worth *p* of the probability, then
  ignore the rest

Grey bars are the tokens your settings just threw away.
