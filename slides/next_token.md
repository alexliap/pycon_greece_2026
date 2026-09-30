# It only ever predicts one token

Here is the part that surprises people. The model does not plan a sentence, or
know where it is going. It does exactly one thing:

```python
prompt = "<some_string>"
answer = []
while True:
    logits = model(prompt)  # a score for every token in the vocabulary
    probs = softmax(logits)  # scores → probabilities
    selected_token = pick(probs)  # choose one
    answer.append(selected_token)  # glue it on, go again
    prompt += tokenizer.decode(selected_token)
```

or

```python
MAX_TOKENS = 512
END_TOKEN = <|end_of_text|>
prompt = "<some_string>"
answer = []
while len(answer) =< MAX_TOKENS or selected_token != END_TOKEN:
    logits = model(prompt)
    probs = softmax(logits)
    selected_token = pick(probs)
    answer.append(selected_token)
    prompt += tokenizer.decode(selected_token)
```

Every essay, every function, every answer you have ever seen from an LLM came
out of that loop, one token at a time.

The interesting question is what `pick` should do.
