# What the model outputs

One score per token in the vocabulary, called a **logit**. A softmax turns the
scores into the probabilities below.

**Temperature** divides the logits before the softmax. Low is confident and
repetitive, high is surprising, then incoherent.
