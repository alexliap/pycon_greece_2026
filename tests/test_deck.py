"""Checks on the deck itself, so a broken notebook fails here and not on stage."""

import ast
import json
import re
import sys
import unittest

from pycon_llm import viz
from pycon_llm.common import read_slide
from pycon_llm.settings import PROJECT_ROOT, SLIDES_PATH, STYLES_PATH

MAIN = PROJECT_ROOT / "main.py"
LAYOUT = PROJECT_ROOT / "layouts" / "main.slides.json"

# Written, but deliberately not in the deck right now. Kept on disk so the prose
# is not lost; delete the file or wire up a cell to take one off this list.
PARKED_SLIDES: set[str] = set()

# Cell types that put something on screen.
VISIBLE_TYPES = ("slide", "sub-slide", "fragment")


def _returned_names(cell: ast.FunctionDef) -> set[str]:
    """Names a cell hands to later cells, from its trailing ``return``."""
    for node in cell.body:
        if isinstance(node, ast.Return):
            if node.value is None:
                return set()
            elts = node.value.elts if isinstance(node.value, (ast.Tuple, ast.List)) else [node.value]
            return {e.id for e in elts if isinstance(e, ast.Name)}
    return set()


def _cells() -> list[ast.FunctionDef]:
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(d, ast.Attribute) and d.attr == "cell" for d in node.decorator_list
        )
    ]


def _slides_referenced() -> list[str]:
    names = []
    for cell in _cells():
        for node in ast.walk(cell):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "read_slide"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                names.append(node.args[0].value)
    return names


class TestReadSlide(unittest.TestCase):
    def test_reads_an_existing_slide(self):
        self.assertIn("LLMs", read_slide("title"))

    def test_missing_slide_names_the_problem(self):
        with self.assertRaises(FileNotFoundError) as caught:
            read_slide("no_such_slide")
        message = str(caught.exception)
        self.assertIn("no_such_slide", message)
        self.assertIn("title", message)  # lists what is available


class TestDeckIntegrity(unittest.TestCase):
    def test_every_referenced_slide_exists(self):
        # main.py once shipped referencing a slides/intro.md that had been
        # deleted, which broke the whole notebook. Never again.
        for name in _slides_referenced():
            with self.subTest(slide=name):
                self.assertTrue(
                    (SLIDES_PATH / f"{name}.md").is_file(),
                    f"main.py reads {name!r} but slides/{name}.md is missing",
                )

    def test_every_slide_file_is_used(self):
        referenced = set(_slides_referenced())
        on_disk = {path.stem for path in SLIDES_PATH.glob("*.md")}
        self.assertEqual(
            on_disk - referenced - PARKED_SLIDES,
            set(),
            "slide files that no cell renders; add to PARKED_SLIDES if deliberate",
        )

    def test_parked_slides_are_really_unused(self):
        # Keeps the allowlist honest: a slide that gets wired back into main.py
        # must be taken off this list.
        referenced = set(_slides_referenced())
        self.assertEqual(
            PARKED_SLIDES & referenced,
            set(),
            "PARKED_SLIDES lists a slide that main.py actually renders",
        )

    def test_no_slide_is_rendered_twice(self):
        names = _slides_referenced()
        self.assertEqual(len(names), len(set(names)))

    def test_no_slide_is_empty(self):
        for path in SLIDES_PATH.glob("*.md"):
            with self.subTest(slide=path.stem):
                self.assertTrue(path.read_text(encoding="utf-8").strip())

    def test_layout_matches_cell_count(self):
        # marimo silently mis-renders the deck when these drift apart.
        layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
        self.assertEqual(len(layout["data"]["cells"]), len(_cells()))

    def test_layout_marks_every_rendering_cell_as_a_slide(self):
        layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
        for index, (cell, entry) in enumerate(zip(_cells(), layout["data"]["cells"])):
            body = [s for s in cell.body if not isinstance(s, ast.Return)]
            renders = bool(body) and isinstance(body[-1], ast.Expr)
            with self.subTest(cell=index):
                self.assertEqual(
                    entry.get("type") in VISIBLE_TYPES,
                    renders,
                    f"cell {index}: layout and notebook disagree about being a slide",
                )

    def test_layout_uses_only_types_marimo_understands(self):
        layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
        allowed = {"slide", "sub-slide", "fragment", "skip"}
        for index, entry in enumerate(layout["data"]["cells"]):
            kind = entry.get("type")
            with self.subTest(cell=index):
                self.assertTrue(kind is None or kind in allowed, f"bad type {kind!r}")

    def test_every_cell_input_is_defined_by_an_earlier_cell(self):
        # marimo resolves each cell's parameters against names other cells
        # return; a typo'd or dropped return leaves a NameError that neither
        # `marimo check` nor a syntax check catches. Reproduces marimo's own
        # dataflow graph well enough to catch that class of bug statically.
        defined: set[str] = set()
        for index, cell in enumerate(_cells()):
            params = {a.arg for a in cell.args.args}
            with self.subTest(cell=index):
                self.assertTrue(
                    params <= defined,
                    f"cell {index} uses {sorted(params - defined)}, "
                    "not returned by any earlier cell",
                )
            defined |= _returned_names(cell)

    def test_layout_is_what_the_builder_would_write(self):
        # Guards against hand-editing the generated file and against forgetting
        # to re-run the builder after adding a cell.
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            import build_layout
        finally:
            sys.path.pop(0)
        expected = build_layout.build()
        actual = json.loads(LAYOUT.read_text(encoding="utf-8"))
        self.assertEqual(
            actual,
            expected,
            "layouts/main.slides.json is stale; run: uv run python scripts/build_layout.py",
        )

    def test_nothing_nests_before_its_parent(self):
        # A deck cannot open on a sub-slide or a fragment: both need a
        # top-level slide in front of them to attach to.
        layout = json.loads(LAYOUT.read_text(encoding="utf-8"))
        seen_slide = False
        for index, entry in enumerate(layout["data"]["cells"]):
            kind = entry.get("type")
            if kind == "slide":
                seen_slide = True
            elif kind in ("sub-slide", "fragment"):
                self.assertTrue(seen_slide, f"cell {index} is a {kind} with no parent")

    def test_a_slide_is_never_both_a_sub_slide_and_a_fragment(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        try:
            import build_layout
        finally:
            sys.path.pop(0)
        self.assertEqual(build_layout.SUB_SLIDES & build_layout.FRAGMENTS, frozenset())

    def test_stylesheet_exists_and_is_wired_up(self):
        self.assertTrue((STYLES_PATH / "deck.css").is_file())
        self.assertIn('css_file="styles/deck.css"', MAIN.read_text(encoding="utf-8"))

    def test_stylesheet_defines_every_class_the_charts_use(self):
        css = (STYLES_PATH / "deck.css").read_text(encoding="utf-8")
        for name in ("pl-chip", "pl-bar-fill", "pl-heat-cell", "pl-dot", "pl-table"):
            with self.subTest(css_class=name):
                self.assertIn(f".{name}", css)


class TestViz(unittest.TestCase):
    def test_escapes_html_in_token_text(self):
        markup = viz.token_chips([("<script>", 1)])
        self.assertNotIn("<script>", markup)
        self.assertIn("&lt;script&gt;", markup)

    def test_makes_whitespace_visible(self):
        self.assertIn("·", viz.token_chips([(" the", 1)]))

    def test_bars_marks_filtered_entries(self):
        markup = viz.bars([("a", 0.6), ("b", 0.4)], kept={"a"})
        self.assertIn("is-cut", markup)

    def test_bars_handles_an_empty_series(self):
        self.assertIn("pl", viz.bars([]))

    def test_heatmap_draws_exact_zeros_as_masked(self):
        markup = viz.heatmap([[1.0, 0.0], [0.5, 0.5]], ["a", "b"], ["a", "b"])
        self.assertIn("is-masked", markup)

    def test_scatter_labels_every_point(self):
        markup = viz.scatter([("cat", 0.0, 0.0, "selected"), ("dog", 1.0, 1.0, "")])
        self.assertIn("cat", markup)
        self.assertIn("dog", markup)

    def test_scatter_handles_a_single_point(self):
        self.assertIn("pl-dot", viz.scatter([("solo", 1.0, 1.0, "")]))

    def test_table_keeps_row_order(self):
        markup = viz.table(["n"], [["first"], ["second"], ["third"]])
        self.assertLess(markup.index("first"), markup.index("second"))
        self.assertLess(markup.index("second"), markup.index("third"))

    def test_table_scrolls_only_when_capped(self):
        self.assertNotIn("is-tall", viz.table(["n"], [["a"]]))
        capped = viz.table(["n"], [["a"]], max_height="20rem")
        self.assertIn("is-tall", capped)
        self.assertIn("max-height:20rem", capped)

    def test_table_highlights_the_last_row(self):
        markup = viz.table(["n"], [["a"], ["b"]], highlight_last=1)
        self.assertEqual(markup.count("is-new"), 1)
        # The highlight must land on the final row, not the first.
        self.assertGreater(markup.index("is-new"), markup.index("a"))


class TestBpePanelTable(unittest.TestCase):
    """The merge table is the centrepiece of the live demo; pin its behaviour."""

    def setUp(self):
        from pycon_llm import bpe, demos

        self.bpe = bpe
        self.markup = demos.bpe_panel(bpe.DEFAULT_CORPUS, bpe.MAX_MERGES).text

    def test_shows_every_merge_without_truncating(self):
        expected = len(self.bpe.train_history(self.bpe.DEFAULT_CORPUS, self.bpe.MAX_MERGES).merges)
        # First cell of each row is the merge number; "seen" is numeric too, so
        # anchor on <tr> rather than matching any <td>.
        numbered = re.findall(r"<tr[^>]*><td>(\d+)</td>", self.markup)
        self.assertEqual(numbered, [str(n) for n in range(1, expected + 1)])

    def test_is_scrollable(self):
        self.assertIn("is-tall", self.markup)


if __name__ == "__main__":
    unittest.main()
