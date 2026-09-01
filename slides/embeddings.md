We have numbers now. The IDs are arbitrary, just row numbers in a lookup table.

So the first thing the network does is swap each ID for a **vector**: a list of
a few thousand numbers, learned during training.

A lookup table is just rows, indexed by ID. Row `17752` holds token `17752`'s
vector, nothing more:

| row (token ID) | stored vector |
|---|---|
| `0` | `[ 0.02, -0.31,  0.77, ...]` |
| ⋮ | ⋮ |
| **`17752`** | **`[ 0.21, -1.04,  0.88, ...]`** |
| ⋮ | ⋮ |
| `50256` | `[ 0.09,  0.60, -0.55, ...]` |

Vocabulary rows deep, embedding dimensions wide. Fetching a vector is one
array index: `table[token_id]`.

The useful part is *where those vectors sit relative to each other*. Words used
in similar contexts end up in similar places, and that turns out to be enough
to make meaning arithmetic.
