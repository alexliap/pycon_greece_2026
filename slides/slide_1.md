# What does an LLM consist of?

Two parts are necessary for an LLM to function, and only one of them is a
neural network.

**1. Tokenizer**: turns text into integers

- A vocabulary plus a list of merge rules. No maths, nothing learned at runtime

**2. Neural Network**: turns those integers into a guess

- Produces one score for every entry in the vocabulary: a vote for what comes next

<br>

> The tokenizer is the small, unglamorous half. It is also the half that decides
> what the model can perceive at all, so that is where we start.
