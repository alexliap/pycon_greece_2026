import unittest

import numpy as np

from pycon_llm import attention


class TestAttention(unittest.TestCase):
    def test_rows_sum_to_one(self):
        for sentence in attention.SENTENCES:
            matrix = attention.attention(sentence)
            np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-9)

    def test_is_strictly_causal(self):
        # Nothing may look forward -- masked cells must be exactly zero, not a
        # small epsilon, because the heatmap draws exact zeros as "forbidden".
        for sentence in attention.SENTENCES:
            matrix = attention.attention(sentence)
            self.assertTrue((np.triu(matrix, k=1) == 0).all())

    def test_first_token_only_sees_itself(self):
        matrix = attention.attention(attention.DEFAULT_SENTENCE)
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0)

    def test_no_negative_weights(self):
        matrix = attention.attention(attention.DEFAULT_SENTENCE)
        self.assertTrue((matrix >= 0).all())

    def test_shape_matches_token_count(self):
        sentence = attention.DEFAULT_SENTENCE
        n = len(attention.tokenize(sentence))
        self.assertEqual(attention.attention(sentence).shape, (n, n))

    def test_empty_sentence_does_not_raise(self):
        self.assertEqual(attention.attention("").shape, (0, 0))

    def test_resolves_every_declared_link(self):
        # This is the payoff of the attention slide: each prepared sentence has
        # a pronoun that must resolve to the right noun.
        for sentence, meta in attention.SENTENCES.items():
            matrix = attention.attention(sentence)
            tokens = attention.tokenize(sentence)
            for source, target in meta["links"].items():
                row = matrix[source].copy()
                row[source] = 0.0
                self.assertEqual(
                    int(np.argmax(row)),
                    target,
                    f"in {sentence!r}: {tokens[source]!r} should point at {tokens[target]!r}",
                )

    def test_declared_links_are_in_range_and_backwards(self):
        for sentence, meta in attention.SENTENCES.items():
            tokens = attention.tokenize(sentence)
            for source, target in meta["links"].items():
                self.assertLess(source, len(tokens))
                self.assertLess(target, source, f"{sentence!r} links forwards")

    def test_unrelated_word_has_no_strong_link(self):
        # "sat" has no declared link, no repeat, and no embedded meaning near
        # anything before it -- its strongest (non-self) link should be weak,
        # not confidently pointing anywhere.
        sentence = "The cat sat on the mat and it purred"
        links = attention.strongest_links(sentence)
        tokens = attention.tokenize(sentence)
        _, _, weight = links[tokens.index("sat") - 1]
        self.assertLess(weight, 0.3)

    def test_repeated_word_attends_to_earlier_occurrence(self):
        sentence = "Maria drank her coffee and ate her bread in Athens"
        links = attention.strongest_links(sentence)
        tokens = attention.tokenize(sentence)
        # "bread" (index 7) has no declared link but shares meaning with "coffee".
        source, target, _ = links[tokens.index("bread") - 1]
        self.assertEqual(source, "bread")
        self.assertEqual(target, "coffee")


if __name__ == "__main__":
    unittest.main()
