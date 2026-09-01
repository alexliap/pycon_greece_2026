# How do we know if it's any good?

**Exact match:** did it produce the expected string?
Cheap and objective, and it fails the moment there are two correct answers.

**Multiple choice:** probability each option, take the highest.
This is what most public benchmarks actually do. It measures recognition, which
is not the same as being useful.

**Perplexity:** "how surprised was the model by this text?"

No expected answer required. Just show it real text and ask how likely it thought
each word was. Perplexity of 1 means perfect confidence; higher means more
confusion.

Try it below. Feed it a sentence like the ones it was trained on, then feed it
nonsense, and watch the number jump.
