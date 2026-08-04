# One vector per word isn't enough

> I sat on the **river bank**.
> I withdrew cash from the **bank**.

Same token, same ID, same starting vector, completely different meaning.

A fixed lookup table cannot fix this, because the meaning is not in the word. It
is in the **other words around it**.

So the network needs a way for words to *look at each other* and adjust
themselves based on what they find.

That mechanism is **attention**, and it is the single idea that made modern LLMs
work.
