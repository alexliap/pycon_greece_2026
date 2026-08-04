# What does an LLM consist of?

Two parts are necessary for an LLM to function, and only one of them is a
neural network.

**1. Tokenizer**: turns text into integers

- A vocabulary plus a list of merge rules. No maths, nothing learned at runtime
- Completely deterministic: the same text always produces the same numbers
- Typically 50,000–200,000 entries, a few megabytes on disk

**2. Neural Network**: turns those integers into a guess

- Where the heavy lifting happens: billions of numbers, almost all of it linear
  algebra
- The only part that is actually *trained*
- Produces one score for every entry in the vocabulary: a vote for what comes next

<br>

> The tokenizer is the small, unglamorous half. It is also the half that decides
> what the model can perceive at all, so that is where we start.
