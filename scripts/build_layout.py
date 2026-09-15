"""Regenerate layouts/main.slides.json from main.py.

marimo silently mis-renders the deck when the layout's cell list drifts out of
sync with the notebook, so the layout is generated rather than hand-edited.

Run after adding, removing or reordering a cell:

    uv run python scripts/build_layout.py

marimo's slides layout accepts one of these per cell:

    {"type": "slide"}      start a new slide (left/right in the deck)
    {"type": "sub-slide"}  nest under the previous slide (down)
    {"type": "fragment"}   reveal inside the current slide on the next click
    {"type": "skip"}       exclude from the presentation
    {}                     no output of its own; folds into the current slide
"""

import ast
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN = PROJECT_ROOT / "main.py"
LAYOUT = PROJECT_ROOT / "layouts" / "main.slides.json"

# Slides that hang *below* their parent instead of beside it. The tokenization
# deep dive is one vertical run: how BPE works, watching it train, then the real
# tokenizer. Linear navigation still walks through all of them in order.
SUB_SLIDES: frozenset[str] = frozenset(
    {
        "bpe_live",
        "tokenizer_playground",
        "attention_math",
        "why_nondeterministic",
        "evaluation_caveat",
        "appendix_bpe_bytes",
    }
)

# Slides that appear *on top of* the previous one, on the next click, the way a
# PowerPoint build works. Use for a punchline you do not want on screen while
# you are still setting it up.
FRAGMENTS: frozenset[str] = frozenset(
    {
        "bpe_idea",
        "attention",
    }
)

# reveal.js deck options.
#
# verticalAlign is deliberately left at its default ("center"). marimo turns it
# straight into a margin -- "top" becomes `margin: 0 20px auto`, which pins every
# slide to the top of the screen and looks wrong on the short ones. Set it to
# "top" only if the tall demo slides start overflowing.
DECK: dict[str, str] = {"transition": "fade"}


def cells(tree: ast.Module) -> list[ast.FunctionDef]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(d, ast.Attribute) and d.attr == "cell"
            for d in node.decorator_list
        )
    ]


def renders(cell: ast.FunctionDef) -> bool:
    """A cell shows something if its last statement is a bare expression."""
    body = [node for node in cell.body if not isinstance(node, ast.Return)]
    return bool(body) and isinstance(body[-1], ast.Expr)


def slide_name(cell: ast.FunctionDef) -> str | None:
    """The slide a cell renders, taken from its read_slide("...") call."""
    for node in ast.walk(cell):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "read_slide"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            return node.args[0].value
    return None


def build() -> dict:
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    entries = []
    for cell in cells(tree):
        if not renders(cell):
            entries.append({})
            continue
        name = slide_name(cell)
        if name in FRAGMENTS:
            kind = "fragment"
        elif name in SUB_SLIDES:
            kind = "sub-slide"
        else:
            kind = "slide"
        entries.append({"type": kind})
    return {"type": "slides", "data": {"cells": entries, "deck": DECK}}


def main() -> None:
    layout = build()
    LAYOUT.write_text(json.dumps(layout) + "\n", encoding="utf-8")

    entries = layout["data"]["cells"]
    counts = {kind: 0 for kind in ("slide", "sub-slide", "fragment")}
    for entry in entries:
        kind = entry.get("type")
        if kind in counts:
            counts[kind] += 1
    print(f"wrote {LAYOUT.relative_to(PROJECT_ROOT)}")
    print(
        f"  {len(entries)} cells: {counts['slide']} slides, "
        f"{counts['sub-slide']} sub-slides, {counts['fragment']} fragments, "
        f"{len(entries) - sum(counts.values())} without output"
    )

    rendered = {
        slide_name(c) for c in cells(ast.parse(MAIN.read_text(encoding="utf-8")))
    }
    for label, names in (("SUB_SLIDES", SUB_SLIDES), ("FRAGMENTS", FRAGMENTS)):
        unknown = names - rendered
        if unknown:
            raise SystemExit(f"{label} names no cell renders: {sorted(unknown)}")

    overlap = SUB_SLIDES & FRAGMENTS
    if overlap:
        raise SystemExit(
            f"a slide cannot be both a sub-slide and a fragment: {sorted(overlap)}"
        )


if __name__ == "__main__":
    main()
