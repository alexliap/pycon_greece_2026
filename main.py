import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", layout_file="layouts/main.slides.json")


@app.cell
def _():
    import marimo as mo
    from tiktoken import get_encoding

    from pycon_llm.common import read_slide

    return mo, read_slide, get_encoding


@app.cell
def _(mo, read_slide):
    mo.md(read_slide("title"))
    return


@app.cell
def _(mo, read_slide):
    mo.md(read_slide("intro"))
    return


@app.cell
def _(mo, read_slide, get_encoding):
    # enc = get_encoding("o200k_base")
    # tids = enc.encode("Hello everyone, thank you for attending!")
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


if __name__ == "__main__":
    app.run()
