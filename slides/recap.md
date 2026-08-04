# 0.1

Everything you saw today, in one paragraph:

> Text is cut into **tokens** by an algorithm that just merges frequent pairs.
> Each token becomes a **vector**, and words adjust those vectors by **looking at
> each other**. Stack that a few dozen times and you get a score for every
> possible next token. Pick one. Do it again.

**What we skipped, and what to read next:**

- positional encodings, layer norm, residual streams: the plumbing
- how gradient descent actually updates 175 billion numbers
- mixture-of-experts, quantization, KV caching: how it is made fast
- 3Blue1Brown's neural network series, and Karpathy's *Let's build GPT*

> You are not at 1.0. Nobody is. But **0.1 is where the black box stops being
> black**, and every one of those topics is now something you can reason about.
