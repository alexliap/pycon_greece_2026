"""The interactive panels, ready to drop into a marimo cell.

Each function takes plain values (whatever the sliders are currently set to) and
returns a marimo element. All the thinking lives in the sibling modules -- this
file only arranges results on screen, so the demos stay easy to rearrange while
rehearsing.
"""

import marimo as mo

from . import attention as attn
from . import bpe, embeddings, sampling, tokenizers, viz

# Palette values mirrored from styles/deck.css for the scatter legend swatches.
_SERIES_1 = "#2a78d6"
_SERIES_2 = "#eb6834"
_SERIES_3 = "#1baf7a"


# --------------------------------------------------------------------------
# Tokenization
# --------------------------------------------------------------------------


def bpe_panel(corpus: str, num_merges: int) -> mo.Html:
    """Live BPE training: the merge table, the vocabulary, and a worked example."""
    if not corpus.strip():
        return mo.Html(viz.note("Type some text to train on."))

    full = bpe.train_history(corpus, bpe.MAX_MERGES)
    model = full.take(num_merges)

    # In the order they were learned, so the table reads as the algorithm
    # running: letter pairs first, whole words by the end.
    rows = [
        [
            str(m.rank + 1),
            f"{m.pair[0]} + {m.pair[1]}",
            str(m.count),
            m.token,
            str(m.token_id),
        ]
        for m in model.merges
    ]

    merge_table = viz.table(
        headers=["#", "most frequent pair", "seen", "new token", "id"],
        rows=rows,
        token_column=3,
        # Highlight the merge the slider just added.
        highlight_last=1 if rows else 0,
        max_height="26rem",
    )

    summary = viz.stats(
        [
            ("base tokens (bytes)", f"{bpe.BASE_VOCAB_SIZE}", ""),
            ("merges applied", f"{len(model.merges)}", "accent"),
            ("vocabulary", f"{model.vocab_size}", "warn"),
            ("available merges", f"{len(full.merges)}", ""),
        ]
    )

    counts = bpe.word_counts(corpus)
    corpus_list = mo.md(
        "\n".join(f"- **{count} ×** `{word}`" for word, count in counts)
    )

    return mo.vstack(
        [
            mo.Html(summary),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(f"**The corpus** ({len(counts)} distinct words)"),
                            corpus_list,
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**Merges learned** (in order)"),
                            mo.Html(merge_table),
                        ]
                    ),
                ],
                widths=[0.7, 1.1],
                gap=1.5,
                align="start",
            ),
        ]
    )


def tokenizer_panel(text: str, encoding_name: str) -> mo.Html:
    """Real tiktoken output: coloured chips, IDs, and the cost of the text."""
    encoder = tokenizers.get_encoder(encoding_name)
    pairs = encoder.tokens(text)
    measurements = tokenizers.token_stats(text, encoder)

    summary = viz.stats(
        [
            ("bytes", f"{measurements['bytes']:.0f}", ""),
            ("characters", f"{measurements['chars']:.0f}", ""),
            ("words", f"{measurements['words']:.0f}", ""),
            ("tokens", f"{measurements['tokens']:.0f}", "accent"),
            ("tokens / word", f"{measurements['tokens_per_word']:.2f}", "warn"),
            ("vocabulary", f"{encoder.vocab_size:,}", ""),
        ]
    )

    return mo.vstack(
        [
            mo.Html(summary),
            mo.Html(viz.token_chips(pairs)),
        ]
    )


def tokenizer_comparison(text: str) -> mo.Html:
    """The same text through every encoding, how tokenizers got better over time."""
    rows = []
    for name, label in tokenizers.ENCODINGS.items():
        encoder = tokenizers.get_encoder(name)
        measurements = tokenizers.token_stats(text, encoder)
        rows.append(
            [
                label,
                f"{encoder.vocab_size:,}",
                f"{measurements['tokens']:.0f}",
                f"{measurements['tokens_per_word']:.2f}",
                f"{measurements['chars_per_token']:.2f}",
            ]
        )
    return mo.Html(
        viz.table(
            headers=[
                "tokenizer",
                "vocabulary",
                "tokens",
                "tokens / word",
                "chars / token",
            ],
            rows=rows,
        )
    )


def language_cost() -> mo.Html:
    """Greek vs English, in tokens. The slide a Greek audience will remember."""
    pairs = [
        ("English", "Good morning Athens, welcome to PyCon!"),
        ("Greek", "Καλημέρα Αθήνα, καλώς ήρθατε στο PyCon!"),
    ]
    rows = []
    for name, encoding_name in [
        ("GPT-2 (2019)", "gpt2"),
        ("GPT-4 (2022)", "cl100k_base"),
        ("GPT-4o (2024)", "o200k_base"),
    ]:
        encoder = tokenizers.get_encoder(encoding_name)
        counts = {
            language: tokenizers.token_stats(text, encoder)["tokens"]
            for language, text in pairs
        }
        ratio = counts["Greek"] / counts["English"] if counts["English"] else 0
        rows.append(
            [
                name,
                f"{counts['English']:.0f}",
                f"{counts['Greek']:.0f}",
                f"{ratio:.1f}×",
            ]
        )
    table = viz.table(
        headers=["tokenizer", "English tokens", "Greek tokens", "Greek costs"],
        rows=rows,
    )
    # Bigger type than the shared .pl-table default; this is the one table on
    # the token_gotchas slide, so it can afford to run larger.
    return mo.Html(f'<div class="language-cost-table">{table}</div>')


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------


def analogy_panel(a: str, b: str, c: str) -> mo.Html:
    """``a - b + c`` drawn in the space, with the winning word called out."""
    results = embeddings.analogy(a, b, c)
    target = embeddings.analogy_target(a, b, c)
    winner = results[0][0]

    points = []
    for other in embeddings.WORDS:
        x, y = embeddings.VECTORS[other]
        role = (
            "selected"
            if other in (a, b, c)
            else ("neighbour" if other == winner else "")
        )
        points.append((other, x, y, role))
    points.append((f"{a} − {b} + {c}", float(target[0]), float(target[1]), "target"))

    chart = viz.scatter(
        points,
        legend=[
            ("words in the sum", _SERIES_1),
            ("the computed point", _SERIES_2),
            ("closest real word", _SERIES_3),
        ],
    )

    answer = mo.md(
        f"### `{a}` − `{b}` + `{c}` = **`{winner}`**\n\n"
        f"The arithmetic lands at {tuple(round(float(v), 2) for v in target)}, and "
        f"the closest actual word to that point is **{winner}**."
    )

    runners = viz.table(
        headers=["candidate", "distance"],
        rows=[[name, f"{distance:.2f}"] for name, distance in results],
    )

    return mo.vstack(
        [
            answer,
            mo.hstack(
                [mo.Html(chart), mo.Html(runners)],
                widths=[2.0, 0.8],
                gap=1.5,
                align="start",
            ),
        ]
    )


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------


def attention_panel(sentence: str, head: str, layer: int, show_values: bool) -> mo.Html:
    """The attention heatmap, plus the strongest link found in each row."""
    tokens = attn.tokenize(sentence)
    matrix = attn.attention(sentence, head, layer).tolist()

    chart = viz.heatmap(
        matrix,
        row_labels=tokens,
        col_labels=tokens,
        show_values=show_values,
        title="Row = the word doing the looking · Column = the word being looked at",
    )

    links = attn.strongest_links(sentence, head, layer)
    rows = [[source, "→", target, f"{weight:.2f}"] for source, target, weight in links]

    note = attn.SENTENCES.get(sentence, {}).get("note", "")
    description = attn.HEADS.get(head, "")

    return mo.vstack(
        [
            mo.md(f"**{description}**" + (f"  \n{note}" if note else "")),
            mo.hstack(
                [
                    mo.Html(chart),
                    mo.vstack(
                        [
                            mo.md("**Strongest link per word**"),
                            mo.Html(
                                viz.table(
                                    headers=["word", "", "looks at", "weight"],
                                    rows=rows,
                                )
                            ),
                        ]
                    ),
                ],
                widths=[2.2, 1.0],
                gap=1.5,
                align="start",
            ),
            mo.Html(
                viz.note(
                    "Hatched cells are <b>masked</b>: when predicting a word the model "
                    "is structurally forbidden from seeing anything to its right. That "
                    "constraint is the whole reason an LLM predicts forwards."
                )
            ),
        ]
    )


# --------------------------------------------------------------------------
# Next-token prediction
# --------------------------------------------------------------------------


def next_token_panel(
    model: sampling.ToyLM,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
) -> mo.Html:
    """The probability distribution over the next token, and what the knobs do to it."""
    context = prompt.lower().split()

    raw = model.top_tokens(context, n=14, temperature=temperature)
    filtered = model.distribution(context, temperature, top_k, top_p)
    survivors = {model.vocab[i] for i in range(len(model.vocab)) if filtered[i] > 0}

    chart = viz.bars(
        raw,
        kept=survivors,
        title=f"P(next token | “{prompt}”)",
    )

    used = model.context_used(context)
    context_note = (
        f"matched on the last {len(used)} word(s): <code>{' '.join(used)}</code>"
        if used
        else "no match, falling back to overall word frequency"
    )

    summary = viz.stats(
        [
            ("temperature", f"{temperature:.2f}", "accent"),
            ("tokens kept", f"{len(survivors)}", ""),
            ("of vocabulary", f"{len(model.vocab)}", ""),
            ("uncertainty (bits)", f"{sampling.entropy(filtered):.2f}", "warn"),
        ]
    )

    return mo.vstack(
        [
            mo.Html(summary),
            mo.Html(chart),
            mo.Html(viz.note(f"Grey bars were cut by top-k / top-p, {context_note}.")),
        ]
    )


def generation_panel(
    model: sampling.ToyLM,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> mo.Html:
    """Run the loop a few times to show what the settings feel like in practice."""
    rows = []
    for offset in range(3):
        text = model.generate(
            prompt,
            steps=14,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed + offset,
        )
        rows.append([f"seed {seed + offset}", f"{prompt} {text}"])

    return mo.vstack(
        [
            mo.md("**Three runs with the current settings**"),
            mo.Html(viz.table(headers=["run", "output"], rows=rows, token_column=1)),
            mo.Html(
                viz.note(
                    "Same settings, different seed, different text. Fix the seed and "
                    "the output is perfectly reproducible. The randomness is a "
                    "choice we make, not something the model does to us."
                )
            ),
        ]
    )


def perplexity_panel(model: sampling.ToyLM, text: str) -> mo.Html:
    """How surprised the model is, token by token."""
    scored = model.score(text)
    if not scored:
        return mo.Html(
            viz.note(
                "None of those words are in the toy model's vocabulary, try another sentence."
            )
        )

    score = sampling.perplexity([probability for _, probability in scored])

    chart = viz.bars(
        [(word, probability) for word, probability in scored],
        title="Probability the model assigned to each word it actually saw",
    )

    verdict = (
        "Low perplexity: this looks like the text it was trained on."
        if score < 10
        else "High perplexity: the model finds this text surprising."
    )

    summary = viz.stats(
        [
            ("perplexity", f"{score:.1f}", "accent" if score < 10 else "warn"),
            ("words scored", f"{len(scored)}", ""),
        ]
    )

    return mo.vstack(
        [
            mo.Html(summary),
            mo.Html(chart),
            mo.Html(
                viz.note(
                    f"{verdict} Perplexity is roughly “how many words was the model "
                    "torn between, on average?” 1.0 would be perfect confidence."
                )
            ),
        ]
    )
