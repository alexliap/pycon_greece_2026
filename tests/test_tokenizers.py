import unittest

from pycon_llm import tokenizers


class TestEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = tokenizers.get_encoder(tokenizers.DEFAULT_ENCODING)

    def test_pieces_and_ids_line_up(self):
        pairs = self.encoder.tokens("Good morning Athens")
        self.assertEqual(len(pairs), len(self.encoder.encode("Good morning Athens")))

    def test_pieces_reconstruct_the_text(self):
        text = "Good morning Athens, welcome to PyCon!"
        self.assertEqual("".join(self.encoder.decode_pieces(text)), text)

    def test_pieces_reconstruct_greek_too(self):
        text = "Καλημέρα Αθήνα, καλώς ήρθατε στο PyCon!"
        self.assertEqual("".join(self.encoder.decode_pieces(text)), text)

    def test_encoders_are_cached(self):
        self.assertIs(
            tokenizers.get_encoder("cl100k_base"), tokenizers.get_encoder("cl100k_base")
        )

    def test_every_advertised_encoding_loads(self):
        for name in tokenizers.ENCODINGS:
            with self.subTest(encoding=name):
                self.assertGreater(tokenizers.get_encoder(name).vocab_size, 0)

    def test_empty_text_is_safe(self):
        self.assertEqual(self.encoder.tokens(""), [])
        self.assertEqual(tokenizers.token_stats("", self.encoder)["chars_per_token"], 0.0)


class TestStats(unittest.TestCase):
    def test_counts_characters_and_tokens(self):
        encoder = tokenizers.get_encoder()
        measurements = tokenizers.token_stats("hello", encoder)
        self.assertEqual(measurements["chars"], 5)
        self.assertGreater(measurements["tokens"], 0)

    def test_counts_bytes_and_words(self):
        encoder = tokenizers.get_encoder()
        measurements = tokenizers.token_stats("hello there world", encoder)
        self.assertEqual(measurements["words"], 3)
        self.assertEqual(measurements["bytes"], 17)

    def test_greek_costs_two_bytes_per_character(self):
        # ASCII is one byte per char; Greek is two. This is the first half of
        # why Greek is expensive, before tokenization even gets involved.
        encoder = tokenizers.get_encoder()
        english = tokenizers.token_stats("kalimera", encoder)
        greek = tokenizers.token_stats("καλημέρα", encoder)
        self.assertEqual(english["bytes"], english["chars"])
        self.assertEqual(greek["bytes"], greek["chars"] * 2)

    def test_tokens_per_word_is_at_least_one_for_known_words(self):
        encoder = tokenizers.get_encoder()
        measurements = tokenizers.token_stats("the cat sat on the mat", encoder)
        self.assertGreaterEqual(measurements["tokens_per_word"], 1.0)

    def test_greek_needs_more_tokens_per_word(self):
        encoder = tokenizers.get_encoder(tokenizers.DEFAULT_ENCODING)
        english = tokenizers.token_stats("Good morning Athens welcome", encoder)
        greek = tokenizers.token_stats("Καλημέρα Αθήνα καλώς ήρθατε", encoder)
        self.assertGreater(greek["tokens_per_word"], english["tokens_per_word"])

    def test_empty_text_has_zero_rates(self):
        encoder = tokenizers.get_encoder()
        measurements = tokenizers.token_stats("", encoder)
        self.assertEqual(measurements["bytes"], 0)
        self.assertEqual(measurements["words"], 0)
        self.assertEqual(measurements["tokens_per_word"], 0.0)

    def test_greek_costs_more_tokens_than_english(self):
        # The claim slide 9 makes. If a future tokenizer fixes this, the slide
        # needs rewriting -- so the test should fail loudly.
        encoder = tokenizers.get_encoder(tokenizers.DEFAULT_ENCODING)
        english = tokenizers.token_stats("Good morning Athens, welcome to PyCon!", encoder)
        greek = tokenizers.token_stats("Καλημέρα Αθήνα, καλώς ήρθατε στο PyCon!", encoder)
        self.assertGreater(greek["tokens"], english["tokens"])
        self.assertLess(greek["chars_per_token"], english["chars_per_token"])

    def test_example_is_non_empty(self):
        self.assertTrue(tokenizers.EXAMPLE.strip())


class TestWarmCache(unittest.TestCase):
    def test_reports_every_encoding(self):
        # The step you run before travelling; it must cover all three, because
        # there is no fallback if an encoding is missing on stage.
        report = tokenizers.warm_cache()
        self.assertEqual(len(report), len(tokenizers.ENCODINGS))
        for name in tokenizers.ENCODINGS:
            self.assertTrue(any(line.startswith(f"{name}: ok") for line in report), name)

    def test_cache_directory_exists_after_warming(self):
        tokenizers.warm_cache()
        self.assertTrue(tokenizers.TIKTOKEN_CACHE_DIR.is_dir())


if __name__ == "__main__":
    unittest.main()
