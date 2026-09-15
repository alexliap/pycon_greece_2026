# It only ever predicts one token

Here is the part that surprises people. The model does not plan a sentence, or
know where it is going. It does exactly one thing:

```python
answer = []
while True:
    logits = model(tokens)  # a score for every token in the vocabulary
    probs = softmax(logits)  # scores → percentages
    answer.append(pick(probs))  # choose one, glue it on, go again
```

or

```python
MAX_TOKENS = 512
END_TOKEN = <|end_of_text|>
answer = []
while len(answer) =< MAX_TOKENS or selected_token != END_TOKEN:
    logits = model(tokens)
    probs = softmax(logits)
    selected_token = pick(probs)
    answer.append(selected_token)
```

Every essay, every function, every answer you have ever seen from an LLM came
out of that loop, one token at a time.

The interesting question is what `pick` should do.
