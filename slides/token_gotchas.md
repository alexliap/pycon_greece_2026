> **Not an actual slide, more like side noted for previous tokenizer slide.**

# Why this actually matters

**"How many r's are in strawberry?"**

The model never sees `s-t-r-a-w-b-e-r-r-y`. It sees three chunks: `st`, `raw`,
`berry`. Asking it to count letters is like asking you to count the pixels in a
word you just read. It is not stupid; it is *blind at that resolution*.

**And Greek costs more than English.**

The tokenizer learned its merges from a corpus that was overwhelmingly English.
Greek text gets shredded into small pieces, which means:

- more tokens for the same sentence → **you pay more per API call**
- more tokens for the same sentence → **it fills the context window faster**
- fewer whole-word tokens → **generally worse quality**
