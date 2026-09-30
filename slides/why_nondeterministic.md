# Why the same prompt gives different answers

Because `pick` rolls a die.

At `temperature=0` it does not. It takes the most likely token every time, and
the model becomes reproducible. So why does anyone use anything else?

- Always taking the safest token produces flat, looping text (*"the
  cat sat on the mat and the cat sat on the mat"*)
- A little randomness is what we experience as fluency and creativity
