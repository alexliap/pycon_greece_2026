# The whole network

Stack that block a few dozen times and you have a transformer:

- **Embed:** every token becomes a vector
- **Transformer × N:** attention (words look at each other), then a small feed-forward
  network (each word thinks on its own). Repeat.
- **Language Model Head:** turn the final vector into one score per token in the vocabulary

Those last scores are called **logits**: one number for every token the model
knows, saying how good a next token it would be.

> GPT-3 is this, 96 times, with 175 billion numbers in the blanks. The shape of
> the thing is not complicated. The scale is.
