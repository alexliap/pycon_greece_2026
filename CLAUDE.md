# pycon_greece_2026

Interactive marimo slides for the talk "LLMs: from 0 to 0.1". `main.py` is the
notebook, slide prose lives in `slides/*.md`, demo logic in `src/pycon_llm/`.

Never use em dashes anywhere: prose, comments, docstrings, commit messages.

## Deck structure

`layouts/main.slides.json` decides how each cell of `main.py` appears. It is
**generated** by `scripts/build_layout.py`; never hand-edit it.

marimo's slides layout gives each cell one of four types:

| type | behaviour |
|---|---|
| `slide` | a new top-level slide (left / right arrows) |
| `sub-slide` | nests *under* the previous slide (down arrow) |
| `fragment` | appears **on top of** the current slide on the next click |
| `skip` | excluded from the presentation |
| `{}` (no type) | no output of its own; folds into the current slide |

There is no Python API for any of this. The only way to set a type is the
layout file, and the only thing that writes the layout file is the build script.

## How to make a fragment slide

A fragment is a whole **cell** that appears on the next click, like a
PowerPoint build. Use it for a punchline you do not want on screen while you are
still setting it up.

Three steps:

**1. Write the content as its own slide file.** A fragment cannot be part of an
existing cell; it has to be a separate cell, so it needs its own markdown file.

```
slides/bpe_idea.md
```

**2. Add a cell to `main.py`, immediately after the cell it should appear over.**
Order in the file is order in the deck.

```python
@app.cell
def _(mo, read_slide):
    # Revealed on the next click, on top of the tokenization slide.
    mo.md(read_slide("bpe_idea"))
    return
```

**3. Name it in `FRAGMENTS` in `scripts/build_layout.py`, then rebuild.**

```python
FRAGMENTS: frozenset[str] = frozenset({"bpe_idea"})
```

```bash
uv run python scripts/build_layout.py
uv run python -m unittest discover -s tests
```

The build script keys off the `read_slide("...")` name inside each cell, so the
name in `FRAGMENTS` must match the markdown filename without `.md`.

### Rules the tooling enforces

- A name in `FRAGMENTS` or `SUB_SLIDES` must be rendered by some cell, or the
  build script exits with an error.
- A slide cannot be in both `FRAGMENTS` and `SUB_SLIDES`.
- A fragment (or sub-slide) must have a top-level `slide` before it; the deck
  cannot open on one.
- The committed layout must equal what the builder would write. `test_deck.py`
  fails if it is stale, so rebuild after touching cells.

### Gotchas

- Fragments are **cell**-level. To reveal three bullets one at a time you need
  three cells and three markdown files. Marking up individual elements inside
  one `mo.md` does not work: there is no Python API, and marimo does not re-sync
  reveal.js after a cell renders.
- Splitting a long slide into fragments is also the fix for content overflowing
  the viewport, since only the revealed part is on screen at a time.
- Pressing "next" walks through fragments and sub-slides in order, so a linear
  run of the deck is unaffected by either.

## Other conventions

**A marimo cell cannot read `.value` of a UI element it defines.** Put
`mo.ui.*` elements in their own cell and read them in the next one. Definition
cells produce no output, so the builder marks them `{}`.

**`.markdown` is `display: contents`.** In `styles/deck.css`, only inherited
properties (font-size, colour) apply to it. `width`, `margin` and `max-width`
are silently ignored; style `.reveal .slides section` instead.

**Theme off marimo, not the OS.** Use `html.dark` / `[data-theme="dark"]`
selectors, never `@media (prefers-color-scheme: dark)`. marimo's theme is
independent of the OS setting, and keying off the OS paints dark colours onto a
light slide.

## Checks

```bash
uv run python scripts/build_layout.py     # after adding/removing/reordering cells
uv run python -m unittest discover -s tests -v
uv run marimo check main.py
uv run marimo run main.py                 # present
```
