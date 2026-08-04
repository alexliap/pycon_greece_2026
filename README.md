# LLMs: from 0 to 0.1

Interactive [marimo](https://marimo.io) slides for Alexandros Liapatis' PyCon
Greece 2026 talk.

Twenty-one slides and five live demos that build intuition for what is inside
an LLM (tokenization, embeddings, attention, next-token prediction, training
and evaluation) in about thirty minutes.

## Run the deck

```bash
uv run marimo edit main.py     # author / rehearse
uv run marimo run main.py      # present (read-only, no code visible)
```

Press `.` or use marimo's slide view to go full screen.

## What is real and what is simulated

The deck contains **no model weights**. Nothing is downloaded while presenting,
provided the tokenizer cache is warm (see below).

| Demo | Real or simulated |
|------|-------------------|
| BPE training | **Real.** The actual algorithm, trained live on whatever you type. See `src/pycon_llm/bpe.py` |
| Tokenizer playground | **Real.** Genuine `tiktoken` encodings (GPT-2, GPT-4, GPT-4o) and their true token IDs |
| Embeddings & analogies | Simulated. ~40 hand-placed 2D vectors, arranged so `king − man + woman == queen` is exact |
| Attention heatmaps | Simulated. Hand-written heads reproducing patterns found in real transformers, with a genuine causal mask |
| Next-token & sampling | Simulated model, **real maths.** A trigram count model supplies the logits; softmax, temperature, top-k and top-p are the real implementations |

Anything simulated says so on the slide it appears on.

## Before you travel: warm the tokenizer cache

`tiktoken` downloads its BPE ranks the first time it loads an encoding. Run this
once, on a good connection, to fill the local cache:

```bash
uv run python -c "from pycon_llm.tokenizers import warm_cache; print(*warm_cache(), sep='\n')"
```

That fills `assets/tiktoken_cache/` (~6 MB) and the deck then runs fully offline.

> `assets/` is gitignored, so a fresh clone downloads on first run. There is no
> offline fallback: if the cache is cold **and** there is no network, the
> tokenizer slides will fail. Warm the cache before the talk.

## Layout

```
main.py                     the notebook, one cell per slide
layouts/main.slides.json    generated; which cells are slides
scripts/build_layout.py     regenerates that file from main.py
slides/*.md                 all slide prose, plain markdown
styles/deck.css             projector-friendly type + every chart style
src/pycon_llm/
    bpe.py                  byte-pair encoding, trained live
    tokenizers.py           tiktoken wrapper + local cache
    embeddings.py           the toy 2D word space
    attention.py            simulated attention heads
    sampling.py             softmax, temperature, top-k, top-p, perplexity
    viz.py                  hand-rolled HTML/CSS charts (no plotting library)
    demos.py                the panels main.py drops into cells
tests/                      96 unittest tests, no marimo import needed
```

Editing prose means editing a markdown file in `slides/`, no Python involved.

### Deck structure

Most slides sit side by side, but the tokenization deep dive nests underneath
its parent, so the deck reads as an outline rather than one long queue:

```
Tokenization
  ↳ Let's train one         (live BPE demo)
  ↳ The real thing          (tiktoken playground)
Why this actually matters
```

Pressing "next" still walks through every slide in order, so a linear
presentation is unaffected. Which slides nest is declared in `SUB_SLIDES` in
`scripts/build_layout.py`.

### Adding a slide

1. write `slides/my_slide.md`
2. add a cell to `main.py` ending in `mo.md(read_slide("my_slide"))`
3. regenerate the layout and re-run the tests:

```bash
uv run python scripts/build_layout.py
uv run python -m unittest discover -s tests -v
```

`tests/test_deck.py` fails if a referenced slide is missing, if a slide file is
never rendered, or if `layouts/main.slides.json` is stale. Those are the ways
this deck has broken before.

> **marimo gotcha:** a cell cannot read `.value` of a UI element it defines. Put
> `mo.ui.*` elements in their own cell and read them in the next one. Definition
> cells produce no output, so they are marked `{}` rather than `{"type": "slide"}`
> in the layout.

marimo's slides layout accepts `slide`, `sub-slide`, `fragment` and `skip` per
cell, plus `speakerNotes`, and `transition` / `verticalAlign` on the deck.

## Checks

```bash
uv run marimo check main.py
uv run python -m unittest discover -s tests -v
```
