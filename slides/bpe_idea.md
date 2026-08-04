## Byte Pair Encoding

Nobody sat down and designed a vocabulary. We **learn** it from data, with an
algorithm you can hold in your head:

1. encode the text as **UTF-8 bytes** (256 to choose from)
2. count every adjacent pair
3. merge the most frequent pair into one new token
4. repeat

That's it. That is the whole algorithm.

Run it 50 times on English and ` the` becomes a single token. Run it 50,000
times and ` tokenization` does too. The vocabulary *grows into the shape of the
language you fed it*, which is exactly why the next slide matters.
