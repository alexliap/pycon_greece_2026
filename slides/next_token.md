# It only ever predicts one token

Here is the part that surprises people. The model does not plan a sentence, or
know where it is going. It does exactly one thing:

```python
while True:
    logits = model(tokens)        # a score for every token in the vocabulary
    probs  = softmax(logits)      # scores → percentages
    tokens.append(pick(probs))    # choose one, glue it on, go again
```

Every essay, every function, every answer you have ever seen from an LLM came
out of that loop, one token at a time.

The interesting question is what `pick` should do.
