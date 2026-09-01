# But watch what they actually measure

All three are built for **pretrained** models: there is no chat format, just raw
next-token prediction. For an **instruction-tuned** model they can mislead: a
model can give the correct answer the lowest perplexity of any candidate, or
score highest on the right multiple-choice option, and still generate a
different, wrong answer when you actually ask it in chat form. Knowing the
right answer scores best is not the same as producing it.
