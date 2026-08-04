# The real thing

Same algorithm, trained by OpenAI on a very large pile of text. Type anything;
these are the actual token IDs a GPT model would receive.

Things worth trying:

- **Greek vs English:** watch the token count, not the character count
- **Numbers:** `1234567` does not split the way you would expect
- **Code:** indentation costs tokens
- **Emoji:** one emoji is often several tokens
