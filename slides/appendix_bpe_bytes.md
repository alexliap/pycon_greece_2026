# Why the vocabulary starts at 256

Unicode defines about **155,000** characters, so 256 is not the number of
characters. It is the size of the **alphabet they are spelled with**.

UTF-8 is *variable length*: one character becomes one to four bytes.

| character | bytes | UTF-8 |
|---|---|---|
| `A` | 1 | `41` |
| `α` | 2 | `CE B1` |
| `あ` | 3 | `E3 81 82` |
| `🐍` | 4 | `F0 9F 90 8D` |

So the base vocabulary is IDs `0` to `255`, and every merge claims the next
free ID: `256`, `257`, `258`, ...

That kills the **out-of-vocabulary** problem for good. Anything writable is
writable as bytes, so a word the tokenizer has never seen is never
*unknown*, only **expensive**: it falls back to more, smaller tokens.
