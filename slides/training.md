# How it learns, in three beats

**1. Pretraining:** read the internet, predict the next token, adjust every
weight slightly when you are wrong. Repeat a few trillion times.

> Result: something that knows an enormous amount about language, and will happily
> continue your question with *another question*, because that is what text does.

**2. Supervised fine-tuning:** show it thousands of examples of
`instruction → good answer`.

> Result: something that answers instead of continuing. This is the step that
> turned a text predictor into an assistant.

**3. Preference tuning (RL):** humans rank pairs of answers, and the model is
nudged toward the ones people preferred.

> Result: something that is helpful and hard to provoke. Manners, essentially.

The base model gets the knowledge. Steps 2 and 3 get the **behaviour**.
