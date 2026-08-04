## Current state

Using an LLM has never been easier. Create an API key, `pip install`, and three
lines later you have the world's knowledge behind a function call.

We got very comfortable very quickly, and most of us never looked inside:

- What does an LLM actually **consist of**?
- Is text really the input and output of the underlying model?
- And why did the smartest systems on earth spend a year or two insisting that
  **strawberry** has two r's? 🍓

## Why "0 to 0.1"?

We are not building GPT-5 in thirty minutes. We are building **intuition**:
the mental model that makes everything you read afterwards click.

- **0**: an LLM is magic, a black box that somehow answers
- **0.1**: an LLM is a tokenizer, a pile of linear algebra, and a loop

**Where we are going:** tokenization → embeddings → attention → next-token
prediction → how these things get trained → how we know if they are any good.
