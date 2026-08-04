import unittest

import numpy as np

from pycon_llm import attention


class TestAttention(unittest.TestCase):
    def test_rows_sum_to_one(self):
        for sentence in attention.SENTENCES:
            for head in attention.HEADS:
                for layer in range(attention.NUM_LAYERS):
                    matrix = attention.attention(sentence, head, layer)
                    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-9)

    def test_is_strictly_causal(self):
        # Nothing may look forward -- masked cells must be exactly zero, not a
        # small epsilon, because the heatmap draws exact zeros as "forbidden".
        for sentence in attention.SENTENCES:
            for head in attention.HEADS:
                for layer in range(attention.NUM_LAYERS):
                    matrix = attention.attention(sentence, head, layer)
                    self.assertTrue(
                        (np.triu(matrix, k=1) == 0).all(),
                        f"{head} leaks into the future on layer {layer}",
                    )

    def test_first_token_only_sees_itself(self):
        matrix = attention.attention(attention.DEFAULT_SENTENCE, "previous", 1)
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0)

    def test_no_negative_weights(self):
        for head in attention.HEADS:
            matrix = attention.attention(attention.DEFAULT_SENTENCE, head, 2)
            self.assertTrue((matrix >= 0).all())

    def test_shape_matches_token_count(self):
        sentence = attention.DEFAULT_SENTENCE
        n = len(attention.tokenize(sentence))
        self.assertEqual(attention.attention(sentence, "sink", 0).shape, (n, n))

    def test_unknown_head_raises(self):
        with self.assertRaises(ValueError):
            attention.attention(attention.DEFAULT_SENTENCE, "nonsense", 0)

    def test_empty_sentence_does_not_raise(self):
        self.assertEqual(attention.attention("", "previous", 0).shape, (0, 0))

    def test_previous_head_looks_backwards_one_step(self):
        links = attention.strongest_links(attention.DEFAULT_SENTENCE, "previous", 2)
        tokens = attention.tokenize(attention.DEFAULT_SENTENCE)
        for index, (source, target, _) in enumerate(links, start=1):
            self.assertEqual(source, tokens[index])
            self.assertEqual(target, tokens[index - 1])

    def test_sink_head_points_at_the_first_token(self):
        links = attention.strongest_links(attention.DEFAULT_SENTENCE, "sink", 3)
        first = attention.tokenize(attention.DEFAULT_SENTENCE)[0]
        self.assertTrue(all(target == first for _, target, _ in links))

    def test_reference_head_resolves_every_declared_link(self):
        # This is the payoff of the attention slide: each prepared sentence has
        # a pronoun that must resolve to the right noun.
        for sentence, meta in attention.SENTENCES.items():
            matrix = attention.attention(sentence, "reference", 3)
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

    def test_deeper_layers_are_more_confident(self):
        sentence = attention.DEFAULT_SENTENCE
        shallow = attention.attention(sentence, "reference", 0)
        deep = attention.attention(sentence, "reference", 3)
        self.assertGreater(deep.max(axis=1).mean(), shallow.max(axis=1).mean())


if __name__ == "__main__":
    unittest.main()
