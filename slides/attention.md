# Attention

Every word asks a question and every word offers an answer:

- **Query:** "what am I looking for?"
- **Key:** "what do I have to offer?"
- **Value:** "here is what you get if you pick me"

Compare each query against every key, turn the scores into percentages, and each
word walks away with a **weighted blend** of the values it cared about. The word
`bank` pulls in a little of `river`, and its vector shifts.

Two details that matter:

- A model runs many attention **heads** side by side, each learning a different
  kind of relationship
- Each word may only look **backwards**, never forwards
