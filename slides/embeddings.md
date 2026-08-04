# From IDs to meaning

We have numbers now. But token `19058` is not *bigger* or *better* than token
`17752`; the IDs are arbitrary, just row numbers in a lookup table.

So the first thing the network does is swap each ID for a **vector**: a list of
a few thousand numbers, learned during training.

```
  17752   →   [0.21, -1.04, 0.88, ... ]
```

The useful part is *where those vectors sit relative to each other*. Words used
in similar contexts end up in similar places, and that turns out to be enough
to make meaning arithmetic.

> Real embeddings have thousands of dimensions. The next slide flattens that to
> 2 so we can look at it. The picture is a cartoon, but the idea is real.
