import marimo

__generated_with = "0.23.16"
app = marimo.App(
    width="full",
    layout_file="layouts/main.slides.json",
    css_file="styles/deck.css",
)


@app.cell
def _():
    import marimo as mo

    from pycon_llm import demos
    from pycon_llm.attention import SENTENCES
    from pycon_llm.bpe import DEFAULT_CORPUS, MAX_MERGES
    from pycon_llm.common import read_slide
    from pycon_llm.embeddings import ANALOGIES
    from pycon_llm.sampling import DEFAULT_PROMPTS, ToyLM
    from pycon_llm.tokenizers import DEFAULT_ENCODING, ENCODINGS, EXAMPLE

    toy_lm = ToyLM.train()
    return (
        ANALOGIES,
        DEFAULT_CORPUS,
        DEFAULT_ENCODING,
        DEFAULT_PROMPTS,
        ENCODINGS,
        EXAMPLE,
        MAX_MERGES,
        SENTENCES,
        demos,
        mo,
        read_slide,
        toy_lm,
    )


@app.cell
def _(mo, read_slide):
    mo.md(read_slide("title"))
    return


@app.cell
def _(mo, read_slide):
    mo.md(read_slide("intro"))
    return


@app.cell
def _(mo, read_slide):
    mo.hstack(
        [
            mo.md(read_slide("slide_1")),
            mo.mermaid(
                """
                graph TD
                    llm[LLM]
                    tokenizer[Tokenizer]
                    neural_network[Neural Network]

                    llm --> tokenizer
                    llm --> neural_network
                """
            ),
        ],
        align="center",
        wrap=True,
        gap=1.0,
        widths=[1.8, 0.8],
    )
    return


@app.cell
def _(mo, read_slide):
    mo.md(read_slide("tokenization"))
    return


@app.cell
def _(mo, read_slide):
    # Revealed on the next click, on top of the tokenization slide.
    # See SUB_SLIDES / FRAGMENTS in scripts/build_layout.py.
    mo.md(read_slide("bpe_idea"))
    return


@app.cell
def _(MAX_MERGES, mo):
    bpe_merges = mo.ui.number(start=0, stop=MAX_MERGES, step=1, value=0, label="Merges")
    return (bpe_merges,)


@app.cell
def _(DEFAULT_CORPUS, bpe_merges, demos, mo, read_slide):
    mo.vstack(
        [
            mo.md(read_slide("bpe_live")),
            mo.md(
                "Drag the slider and watch it happen:\n\n"
                "- `▁` stands for the space byte, so the space really "
                "is part of the token\n"
                "- The first merges are dull letter pairs. `we` wins, "
                "appearing 12 times\n"
                "- By merge 4 you have `▁newe`; by merge 10 the whole "
                "word `▁newest`\n"
                "- Nobody chose those. They fell out of counting."
            ),
            bpe_merges,
            demos.bpe_panel(DEFAULT_CORPUS, bpe_merges.value),
        ],
        gap=1.0,
    )

    return


@app.cell
def _(DEFAULT_ENCODING, ENCODINGS, mo):
    tok_encoding = mo.ui.dropdown(
        options={label: name for name, label in ENCODINGS.items()},
        value=ENCODINGS[DEFAULT_ENCODING],
        label="Tokenizer",
    )
    return (tok_encoding,)


@app.cell
def _(EXAMPLE, mo):
    tok_text = mo.ui.text_area(
        value=EXAMPLE, label="Your text", rows=3, full_width=True
    )
    return (tok_text,)


@app.cell
def _(demos, mo, read_slide, tok_encoding, tok_text):
    mo.vstack(
        [
            mo.md(read_slide("tokenizer_playground")),
            tok_encoding,
            tok_text,
            demos.tokenizer_panel(tok_text.value, tok_encoding.value),
            mo.md("**The same text through every generation of tokenizer**"),
            demos.tokenizer_comparison(tok_text.value),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo, read_slide):
    mo.vstack(
        [
            mo.md("# **Embeddings** → From IDs to meaning"),
            mo.mermaid(
                """
                graph LR
                    tokenizer["✅ Tokenizer"]
                    neural_network[Neural Network]

                    tokenizer --> neural_network
                """
            ),
            mo.md(read_slide("embeddings")),
        ],
        gap=0.4,
    )
    return


@app.cell
def _(ANALOGIES, mo):
    analogy_choice = mo.ui.dropdown(
        options={f"{a} - {b} + {c}": (a, b, c) for a, b, c in ANALOGIES},
        value="athens - greece + france",
        label="Try",
    )
    return (analogy_choice,)


@app.cell
def _(analogy_choice, demos, mo, read_slide):
    mo.vstack(
        [
            mo.md(read_slide("analogy_demo")),
            analogy_choice,
            demos.analogy_panel(*analogy_choice.value),
        ],
        gap=0.4,
    )
    return


@app.cell
def _(mo, read_slide):
    mo.md(read_slide("context"))
    return


@app.cell
def _(mo, read_slide):
    # Revealed on the next click, on top of the context slide.
    # See SUB_SLIDES / FRAGMENTS in scripts/build_layout.py.
    mo.md(read_slide("attention"))
    return


@app.cell
def _(mo, read_slide):
    mo.hstack(
        [
            mo.Html(
                '<div class="pl-airy">' + mo.md(read_slide("qkv")).text + "</div>"
            ),
            mo.mermaid(
                """
                graph LR
                    x[Input X]
                    q[Query]
                    k[Key]
                    v[Value]

                    x --> q
                    x --> k
                    x --> v
                """
            ),
        ],
        align="center",
        wrap=True,
        gap=1.0,
        widths=[1.6, 1.0],
    )
    return


@app.cell
def _(SENTENCES, mo):
    attn_sentence = mo.ui.dropdown(
        options=list(SENTENCES), value=next(iter(SENTENCES)), label="Sentence"
    )
    attn_values = mo.ui.checkbox(value=False, label="Show numbers")
    return attn_sentence, attn_values


@app.cell
def _(
    attn_sentence,
    attn_values,
    demos,
    mo,
    read_slide,
):
    mo.vstack(
        [
            mo.md(read_slide("attention_scores")),
            mo.hstack(
                [attn_sentence, attn_values],
                justify="start",
                gap=1.2,
                wrap=True,
            ),
            demos.attention_panel(
                attn_sentence.value,
                attn_values.value,
                masked=False,
                side=mo.Html(
                    '<div class="pl-formula">'
                    + mo.md(
                        r"$$\text{scores} = \operatorname{softmax}\!\left("
                        r"\dfrac{QK^{\top}}{\sqrt{d_k}}\right)$$"
                    ).text
                    + "</div>"
                ),
            ),
        ],
        gap=0.5,
    )
    return


@app.cell
def _(
    attn_sentence,
    attn_values,
    demos,
    mo,
    read_slide,
):
    mo.vstack(
        [
            mo.md(read_slide("attention_demo")),
            mo.hstack(
                [attn_sentence, attn_values],
                justify="start",
                gap=1.2,
                wrap=True,
            ),
            demos.attention_panel(
                attn_sentence.value,
                attn_values.value,
                side=mo.Html(
                    '<div class="pl-formula">'
                    + mo.md(
                        r"$$\text{scores} = \operatorname{softmax}\!\left("
                        r"\dfrac{QK^{\top}}{\sqrt{d_k}} + M\right)$$"
                    ).text
                    + '<div class="pl-formula-where">'
                    + mo.md(r"where $M_{ij} = -\infty$ if $j > i$, else $0$").text
                    + "</div></div>"
                ),
            ),
        ],
        gap=0.5,
    )
    return


@app.cell
def _(mo, read_slide):
    mo.vstack(
        [
            mo.md(read_slide("attention_output")),
            mo.Html(
                '<div class="pl-formula">'
                + mo.md(
                    r"$$\text{Attention}(Q,K,V) = \operatorname{softmax}\!\left("
                    r"\dfrac{QK^{\top}}{\sqrt{d_k}} + M\right)V$$"
                ).text
                + "</div>"
            ),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo, read_slide):
    mo.hstack(
        [
            mo.md(read_slide("attention_heads")),
            mo.mermaid(
                """
                graph TD
                    x[Input X]
                    h1["Head 1: Q,K,V → Attention"]
                    h2["Head 2: Q,K,V → Attention"]
                    hdots["..."]
                    concat[Concat]
                    wo[W_O]
                    out[Output]

                    x --> h1
                    x --> h2
                    x --> hdots
                    h1 --> concat
                    h2 --> concat
                    hdots --> concat
                    concat --> wo --> out
                """
            ),
        ],
        align="center",
        wrap=True,
        gap=1.0,
        widths=[1.6, 1.0],
    )
    return


@app.cell
def _(mo, read_slide):
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md(read_slide("feed_forward")),
                    mo.Html(
                        '<div class="pl-formula">'
                        + mo.md(
                            r"$$\text{FFN}(x) = \text{activation}(xW_1 + b_1)\,W_2 + b_2$$"
                        ).text
                        + "</div>"
                    ),
                ],
                gap=1.0,
            ),
            mo.mermaid(
                """
                graph TD
                    x[Word vector x]
                    w1[Matrix W1]
                    act[Activation]
                    w2[Matrix W2]
                    out[New word vector]

                    x --> w1 --> act --> w2 --> out
                """
            ),
        ],
        align="center",
        wrap=True,
        gap=1.0,
        widths=[1.6, 1.0],
    )
    return


@app.cell
def _(mo, read_slide):
    mo.hstack(
        [
            mo.Html(
                '<div class="pl-airy">' + mo.md(read_slide("transformer_block")).text + "</div>"
            ),
            mo.mermaid(
                """
                graph TD
                    x[Input]
                    attn[Multi-Head Attention]
                    ff[Feed-Forward]
                    out[Output]

                    x --> attn --> ff --> out
                """
            ),
        ],
        align="center",
        wrap=True,
        gap=1.0,
        widths=[1.6, 1.0],
    )
    return


@app.cell
def _(mo, read_slide):
    mo.hstack(
        [
            mo.md(read_slide("transformer_stack")),
            mo.mermaid(
                """
                graph TD
                    tokens[Token IDs]
                    embed[Embedding]
                    block1[Attention + Feed-forward / Transformer]
                    blockn[... x N blocks]
                    logits[LM Head -> Logits: one score per token]

                    tokens --> embed
                    embed --> block1
                    block1 --> blockn
                    blockn --> logits
                """
            ),
        ],
        align="center",
        wrap=True,
        gap=1.0,
        widths=[1.6, 1.0],
    )
    return


@app.cell
def _(mo, read_slide):
    mo.md(read_slide("next_token"))
    return


@app.cell
def _(DEFAULT_PROMPTS, mo):
    gen_prompt = mo.ui.dropdown(
        options=list(DEFAULT_PROMPTS), value="the cat", label="Prompt"
    )
    gen_temperature = mo.ui.slider(
        start=0.1, stop=2.0, step=0.1, value=1.0, label="Temperature", show_value=True
    )
    gen_seed = mo.ui.number(start=0, stop=999, value=1, label="Seed")
    return gen_prompt, gen_seed, gen_temperature


@app.cell
def _(
    demos,
    gen_prompt,
    gen_temperature,
    mo,
    read_slide,
    toy_lm,
):
    mo.vstack(
        [
            mo.md(read_slide("sampling_demo")),
            mo.hstack(
                [gen_prompt, gen_temperature],
                justify="start",
                gap=1.2,
                wrap=True,
            ),
            demos.next_token_panel(
                toy_lm,
                gen_prompt.value,
                gen_temperature.value,
                side=mo.Html(
                    '<div class="pl-formula">'
                    + mo.md(
                        r"$$\operatorname{softmax}\!\left(\dfrac{\text{logits}}{T}\right)$$"
                    ).text
                    + "</div>"
                ),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(
    mo,
    read_slide,
):
    mo.md(read_slide("why_nondeterministic"))

    return


@app.cell
def _(mo, read_slide):
    mo.hstack(
        [
            mo.md(read_slide("training")),
            mo.mermaid(
                """
                graph TD
                    corpus[Trillions of tokens]
                    base[Base model: predicts text]
                    sft[Instruction tuned: answers questions]
                    rl[Preference tuned: helpful and safe]

                    corpus -->|Pretraining| base
                    base -->|Supervised fine-tuning| sft
                    sft -->|Reinforcement Learning| rl
                """
            ),
        ],
        align="center",
        wrap=True,
        gap=1.0,
        widths=[1.7, 1.0],
    )
    return


@app.cell
def _(mo):
    eval_text = mo.ui.text(
        value="the model reads the text and predicts the next word",
        label="Score this text",
        full_width=True,
    )
    return (eval_text,)


@app.cell
def _(demos, eval_text, mo, read_slide, toy_lm):
    mo.vstack(
        [
            mo.md(read_slide("evaluation")),
            eval_text,
            demos.perplexity_panel(toy_lm, eval_text.value),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(demos, mo, read_slide):
    # Nested under the evaluation slide (down arrow), so the metrics get their
    # own beat before the caveat about what they miss.
    # See SUB_SLIDES / FRAGMENTS in scripts/build_layout.py.
    mo.vstack(
        [
            mo.md(read_slide("evaluation_caveat")),
            demos.eval_mismatch_panel(),
        ],
        gap=1.0,
    )
    return


@app.cell
def _(mo, read_slide):
    mo.md(read_slide("thanks"))
    return


@app.cell
def _(mo, read_slide):
    mo.md(read_slide("appendix"))
    return


@app.cell
def _(mo, read_slide):
    # Nested under the appendix slide (down arrow): the byte/UTF-8 detail cut
    # from the live BPE slide for time.
    # See SUB_SLIDES / FRAGMENTS in scripts/build_layout.py.
    mo.md(read_slide("appendix_bpe_bytes"))
    return


@app.cell
def _(mo, read_slide):
    # Nested under the appendix_bpe_bytes slide (down arrow): the QKV/
    # multi-head matrix detail cut from the live attention slide for time.
    # See SUB_SLIDES / FRAGMENTS in scripts/build_layout.py.
    mo.hstack(
        [
            mo.md(read_slide("attention_math")),
            mo.vstack(
                [
                    mo.md("**Attention**"),
                    mo.mermaid(
                        """
                        graph LR
                            x[Input X]
                            wq[W_Q]
                            wk[W_K]
                            wv[W_V]
                            q[Q]
                            k[K]
                            v[V]
                            scores["Q · Kᵀ"]
                            softmax[softmax]
                            out[Weighted output]

                            x --> wq --> q
                            x --> wk --> k
                            x --> wv --> v
                            q --> scores
                            k --> scores
                            scores --> softmax
                            softmax --> out
                            v --> out
                        """
                    ),
                    mo.md("**Multi-head attention**"),
                    mo.mermaid(
                        """
                        graph TD
                            x[Input X]
                            h1["Head 1: Q,K,V → Attention"]
                            h2["Head 2: Q,K,V → Attention"]
                            hdots["..."]
                            concat[Concat]
                            wo[W_O]
                            out[Output]

                            x --> h1
                            x --> h2
                            x --> hdots
                            h1 --> concat
                            h2 --> concat
                            hdots --> concat
                            concat --> wo --> out
                        """
                    ),
                ],
                gap=1.0,
            ),
        ],
        align="center",
        wrap=True,
        gap=2.5,
        widths=[1.0, 1.3],
    )
    return


if __name__ == "__main__":
    app.run()
