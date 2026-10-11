## Tokenization

A neural network cannot read. It multiplies matrices of numbers, and that is
genuinely all it does. So text has to become numbers first.

The only real question is **where to cut**. There are three logical answers:

| Cut by | `tokenization` becomes | Vocabulary | Upside | Drawback |
|:---:|:---:|:---:|:---:|:---:|
| **Character** | `t` `o` `k` `e` `n` … (12 tokens) | short | Nothing is ever unknown | Very long sequences |
| **Word** | `tokenization` (1 token) | unbounded | Short sequences | existence of unknown words |
| **Subword** | `token` `ization` (2 tokens) | fixed | Frequent words stay whole, rare ones fall back to pieces | The split must be **learned from data**, so it inherits that data's biases |

Everyone uses **subwords**. The interesting question is how you find them.
