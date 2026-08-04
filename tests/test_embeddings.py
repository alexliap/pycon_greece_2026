import unittest

from pycon_llm import embeddings


class TestEmbeddings(unittest.TestCase):
    def test_every_themed_word_has_a_vector(self):
        for theme, members in embeddings.THEMES.items():
            for word in members:
                self.assertIn(word, embeddings.VECTORS, f"{word} ({theme}) has no vector")

    def test_every_vector_has_a_theme(self):
        for word in embeddings.WORDS:
            self.assertNotEqual(embeddings.theme_of(word), "other", word)

    def test_unknown_word_raises_clearly(self):
        with self.assertRaises(KeyError):
            embeddings.vector("definitely-not-a-word")

    def test_nearest_neighbour_shares_the_theme(self):
        # The clusters are hand-placed; if an edit breaks one, this catches it.
        for word in embeddings.WORDS:
            closest = embeddings.nearest(word, 1)[0][0]
            self.assertEqual(
                embeddings.theme_of(closest),
                embeddings.theme_of(word),
                f"{word}'s nearest neighbour {closest} is in another theme",
            )

    def test_nearest_excludes_the_word_itself(self):
        self.assertNotIn("cat", [w for w, _ in embeddings.nearest("cat", 5)])

    def test_nearest_returns_requested_count(self):
        self.assertEqual(len(embeddings.nearest("cat", 6)), 6)

    def test_nearest_is_sorted_by_distance(self):
        distances = [d for _, d in embeddings.nearest("wine", 5)]
        self.assertEqual(distances, sorted(distances))

    def test_king_man_woman_is_queen(self):
        self.assertEqual(embeddings.analogy("king", "man", "woman")[0][0], "queen")

    def test_every_advertised_analogy_lands(self):
        # These are the options on the slide's dropdown -- all must be exact.
        expected = {
            ("king", "man", "woman"): "queen",
            ("queen", "woman", "man"): "king",
            ("athens", "greece", "france"): "paris",
            ("paris", "france", "italy"): "rome",
            ("kitten", "cat", "dog"): "puppy",
            ("puppy", "dog", "cat"): "kitten",
        }
        for triple in embeddings.ANALOGIES:
            self.assertIn(triple, expected, f"{triple} has no expected answer")
            self.assertEqual(embeddings.analogy(*triple)[0][0], expected[triple], triple)

    def test_analogy_excludes_its_own_inputs(self):
        results = [w for w, _ in embeddings.analogy("king", "man", "woman")]
        for word in ("king", "man", "woman"):
            self.assertNotIn(word, results)

    def test_capital_offset_is_consistent(self):
        # The country -> capital direction must be identical for every pair,
        # which is what makes the analogy exact rather than approximate.
        pairs = [("greece", "athens"), ("france", "paris"), ("italy", "rome"), ("japan", "tokyo")]
        offsets = {
            tuple(round(float(v), 6) for v in embeddings.vector(city) - embeddings.vector(country))
            for country, city in pairs
        }
        self.assertEqual(len(offsets), 1, f"inconsistent capital offsets: {offsets}")

    def test_gender_offset_is_consistent(self):
        pairs = [("man", "woman"), ("king", "queen"), ("prince", "princess"), ("boy", "girl")]
        offsets = {
            tuple(round(float(v), 6) for v in embeddings.vector(f) - embeddings.vector(m))
            for m, f in pairs
        }
        self.assertEqual(len(offsets), 1, f"inconsistent gender offsets: {offsets}")


if __name__ == "__main__":
    unittest.main()
