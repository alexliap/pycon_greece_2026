import unittest

from pycon_llm import bpe


class TestBPE(unittest.TestCase):
    def test_first_merge_is_the_most_frequent_pair(self):
        # "ab" appears three times; nothing else repeats as often.
        model = bpe.train("ab ab ab cd", num_merges=1)
        self.assertEqual(len(model.merges), 1)
        self.assertEqual(model.merges[0].token, "▁a")

    def test_merge_counts_are_weighted_by_word_frequency(self):
        model = bpe.train("lo lo lo lo hi", num_merges=3)
        first = model.merges[0]
        self.assertEqual(first.count, 4)

    def test_history_is_prefix_stable(self):
        # The slider slices a single training run, so training with N merges
        # must equal taking the first N of a longer run. Only meaningful up to
        # MAX_MERGES, which is where the slider stops.
        full = bpe.train_history(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES)
        for n in (0, 1, 5, bpe.MAX_MERGES):
            with self.subTest(merges=n):
                self.assertEqual(
                    bpe.train(bpe.DEFAULT_CORPUS, n).merges, full.take(n).merges
                )

    def test_zero_merges_is_character_level(self):
        model = bpe.train(bpe.DEFAULT_CORPUS, num_merges=0)
        self.assertEqual(model.encode("cat"), ["▁", "c", "a", "t"])

    def test_merging_reduces_token_count(self):
        # Must use a word the corpus actually contains; merges learned from one
        # vocabulary do nothing for unrelated text.
        full = bpe.train_history(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES)
        few = full.take(2)
        many = full.take(bpe.MAX_MERGES)
        self.assertLess(len(many.encode("newest")), len(few.encode("newest")))

    def test_a_corpus_word_collapses_to_one_token(self):
        # The payoff of the live demo: at full merges "newest" is a single token.
        full = bpe.train_history(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES)
        self.assertEqual(full.encode("newest"), ["▁newest"])

    def test_unseen_word_falls_back_to_pieces(self):
        full = bpe.train_history(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES)
        self.assertGreater(len(full.encode("joker")), 1)

    def test_default_corpus_matches_its_word_counts(self):
        self.assertEqual(bpe.word_counts(bpe.DEFAULT_CORPUS), list(bpe.CORPUS.items()))

    def test_word_counts_preserves_first_appearance_order(self):
        self.assertEqual(
            bpe.word_counts("beta alpha beta gamma alpha beta"),
            [("beta", 3), ("alpha", 2), ("gamma", 1)],
        )

    def test_encode_round_trips(self):
        model = bpe.train_history(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES)
        text = "the quick brown fox"
        rejoined = "".join(model.encode(text)).replace(bpe.WORD_START, " ").strip()
        self.assertEqual(rejoined, text)

    def test_unseen_text_still_encodes(self):
        model = bpe.train_history(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES)
        self.assertTrue(model.encode("ζφγ"))

    def test_training_stops_when_nothing_repeats(self):
        model = bpe.train("abcdefg", num_merges=bpe.MAX_MERGES)
        self.assertLess(len(model.merges), bpe.MAX_MERGES)

    def test_vocab_grows_by_one_per_merge(self):
        full = bpe.train_history(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES)
        base = len(full.base_alphabet)
        self.assertEqual(len(full.take(7).vocab), base + 7)

    def test_base_vocabulary_is_every_byte(self):
        self.assertEqual(bpe.BASE_VOCAB_SIZE, 2**8)

    def test_merged_tokens_get_ids_after_the_bytes(self):
        # The claim on the slide: merges take IDs 256, 257, 258, ...
        full = bpe.train_history(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES)
        ids = [m.token_id for m in full.merges]
        self.assertEqual(ids, list(range(256, 256 + len(full.merges))))

    def test_vocab_size_counts_all_bytes_plus_merges(self):
        full = bpe.train_history(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES)
        for n in (0, 1, 7, bpe.MAX_MERGES):
            with self.subTest(merges=n):
                self.assertEqual(full.take(n).vocab_size, bpe.BASE_VOCAB_SIZE + n)

    def test_vocab_size_exceeds_the_symbols_this_corpus_used(self):
        # A real tokenizer carries all 256 bytes whether the corpus used them or
        # not; that is what makes unknown text impossible.
        full = bpe.train_history(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES)
        self.assertGreater(full.vocab_size, len(full.vocab))

    def test_is_deterministic(self):
        a = bpe.train(bpe.DEFAULT_CORPUS, 25)
        b = bpe.train(bpe.DEFAULT_CORPUS, 25)
        self.assertEqual(a.merges, b.merges)

    def test_empty_corpus_does_not_raise(self):
        self.assertEqual(bpe.train("", num_merges=5).merges, ())


if __name__ == "__main__":
    unittest.main()
