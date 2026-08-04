import unittest

import numpy as np

from pycon_llm import sampling


class TestSoftmax(unittest.TestCase):
    def test_sums_to_one(self):
        probs = sampling.softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(float(probs.sum()), 1.0)

    def test_preserves_ordering(self):
        probs = sampling.softmax(np.array([0.5, 3.0, -1.0]))
        self.assertEqual(int(np.argmax(probs)), 1)

    def test_low_temperature_sharpens(self):
        logits = np.array([1.0, 2.0, 3.0])
        cold = sampling.entropy(sampling.softmax(logits, 0.2))
        hot = sampling.entropy(sampling.softmax(logits, 2.0))
        self.assertLess(cold, hot)

    def test_zero_temperature_is_greedy(self):
        probs = sampling.softmax(np.array([1.0, 5.0, 2.0]), temperature=0.0)
        self.assertEqual(float(probs[1]), 1.0)

    def test_handles_large_logits_without_overflow(self):
        probs = sampling.softmax(np.array([1000.0, 1001.0]))
        self.assertAlmostEqual(float(probs.sum()), 1.0)
        self.assertTrue(np.isfinite(probs).all())


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.probs = sampling.softmax(np.array([4.0, 3.0, 2.0, 1.0, 0.0]))

    def test_top_k_keeps_exactly_k(self):
        self.assertEqual(np.count_nonzero(sampling.top_k_filter(self.probs, 2)), 2)

    def test_top_k_keeps_the_best_ones(self):
        filtered = sampling.top_k_filter(self.probs, 2)
        self.assertGreater(filtered[0], 0)
        self.assertEqual(filtered[4], 0)

    def test_top_k_renormalises(self):
        self.assertAlmostEqual(float(sampling.top_k_filter(self.probs, 3).sum()), 1.0)

    def test_top_k_off_is_a_no_op(self):
        np.testing.assert_allclose(sampling.top_k_filter(self.probs, 0), self.probs)

    def test_top_p_keeps_at_least_the_requested_mass(self):
        # Measured against the original distribution, the kept set must cover p.
        filtered = sampling.top_p_filter(self.probs, 0.6)
        kept = self.probs[filtered > 0].sum()
        self.assertGreaterEqual(float(kept), 0.6)

    def test_top_p_drops_the_tail(self):
        self.assertLess(np.count_nonzero(sampling.top_p_filter(self.probs, 0.5)), self.probs.size)

    def test_top_p_one_is_a_no_op(self):
        np.testing.assert_allclose(sampling.top_p_filter(self.probs, 1.0), self.probs)

    def test_top_p_never_empties(self):
        self.assertGreater(np.count_nonzero(sampling.top_p_filter(self.probs, 0.01)), 0)


class TestToyLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lm = sampling.ToyLM.train()

    def test_logits_cover_the_vocabulary(self):
        self.assertEqual(self.lm.next_token_logits(["the"]).shape[0], len(self.lm.vocab))

    def test_learns_a_real_continuation(self):
        best = self.lm.top_tokens(["the", "cat"], n=1)[0][0]
        self.assertEqual(best, "sat")

    def test_uses_the_longest_matching_context(self):
        self.assertEqual(self.lm.context_used(["the", "cat"]), ("the", "cat"))
        self.assertEqual(self.lm.context_used(["zzz", "cat"]), ("cat",))
        self.assertEqual(self.lm.context_used(["zzz"]), ())

    def test_same_seed_gives_same_text(self):
        a = self.lm.generate("the cat", steps=10, seed=42)
        b = self.lm.generate("the cat", steps=10, seed=42)
        self.assertEqual(a, b)

    def test_different_seeds_can_differ(self):
        outputs = {self.lm.generate("the cat", steps=12, seed=s) for s in range(8)}
        self.assertGreater(len(outputs), 1)

    def test_generation_respects_step_budget(self):
        text = self.lm.generate("the cat", steps=5, temperature=0.5, seed=1)
        self.assertLessEqual(len(text.split()), 5)

    def test_greedy_generation_is_stable_across_seeds(self):
        a = self.lm.generate("the cat", steps=8, temperature=0.0, seed=1)
        b = self.lm.generate("the cat", steps=8, temperature=0.0, seed=999)
        self.assertEqual(a, b)

    def test_distribution_is_normalised_under_every_filter(self):
        for temperature in (0.3, 1.0, 1.8):
            for k in (0, 3):
                for p in (0.5, 1.0):
                    probs = self.lm.distribution(["the"], temperature, k, p)
                    self.assertAlmostEqual(float(probs.sum()), 1.0)


class TestPerplexity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lm = sampling.ToyLM.train()

    def test_perfect_confidence_is_one(self):
        self.assertAlmostEqual(sampling.perplexity([1.0, 1.0]), 1.0)

    def test_familiar_text_scores_better_than_nonsense(self):
        familiar = sampling.perplexity(
            [p for _, p in self.lm.score("the cat sat on the mat")]
        )
        nonsense = sampling.perplexity(
            [p for _, p in self.lm.score("python barked athens loudly")]
        )
        self.assertLess(familiar, nonsense)

    def test_score_skips_unknown_words(self):
        scored = self.lm.score("the cat qwertyuiop")
        self.assertNotIn("qwertyuiop", [word for word, _ in scored])


if __name__ == "__main__":
    unittest.main()
