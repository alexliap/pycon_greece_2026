"""Hand-rolled HTML/CSS charts.

No plotting library: every visual in the deck is a handful of divs styled by
``styles/deck.css``. That keeps the dependency list at three packages, renders
identically on any projector, and means nothing can fail to load mid-talk.

Every builder returns an HTML string wrapped in ``<div class="pl">``, which is
the scope the stylesheet's colour tokens are defined on.
"""

from __future__ import annotations

from html import escape

# Sequential ramp is expressed as alpha over the surface colour, so the same
# code reads correctly in light and dark mode -- low values recede toward the
# background, which is exactly what "near zero" should look like.
_HEAT_MIN_ALPHA = 0.04
_HEAT_TEXT_FLIP = 0.55
_HEAT_FONT_MIN = 0.52
_HEAT_FONT_MAX = 0.95


def wrap(inner: str) -> str:
    """Scope some markup to the deck's colour tokens."""
    return f'<div class="pl">{inner}</div>'


def _visible(text: str) -> str:
    """Make whitespace visible -- half the point of the tokenizer slide."""
    return (
        escape(text)
        .replace(" ", "·")
        .replace("\n", "⏎")
        .replace("\t", "→")
    )


def token_chips(
    pairs: list[tuple[str, int]],
    show_ids: bool = True,
    max_tokens: int = 240,
) -> str:
    """Coloured chips, one per token.

    Colour cycles purely to mark boundaries -- it carries no meaning, and the
    gaps plus the printed IDs say the same thing without relying on colour.
    """
    chips = []
    for i, (piece, token_id) in enumerate(pairs[:max_tokens]):
        label = _visible(piece) or "·"
        id_html = f'<span class="pl-chip-id">{token_id}</span>' if show_ids else ""
        chips.append(f'<span class="pl-chip pl-chip-{i % 5}">{label}{id_html}</span>')

    truncated = ""
    if len(pairs) > max_tokens:
        truncated = f'<p class="pl-note">…and {len(pairs) - max_tokens} more tokens.</p>'

    return wrap(f'<div class="pl-chips">{"".join(chips)}</div>{truncated}')


def stats(items: list[tuple[str, str, str]]) -> str:
    """Stat tiles: ``(label, value, style)`` where style is ""/"accent"/"warn"."""
    tiles = []
    for label, value, style in items:
        cls = f" is-{style}" if style else ""
        tiles.append(
            f'<div><div class="pl-stat-value{cls}">{escape(value)}</div>'
            f'<div class="pl-stat-label">{escape(label)}</div></div>'
        )
    return wrap(f'<div class="pl-stats">{"".join(tiles)}</div>')


def bars(
    items: list[tuple[str, float]],
    kept: set[str] | None = None,
    value_format: str = "{:.1%}",
    title: str | None = None,
) -> str:
    """A ranked horizontal bar chart of one series.

    ``kept`` marks which labels survived top-k / top-p; everything else is drawn
    in neutral grey so the effect of the sliders is visible rather than implied.
    """
    if not items:
        return wrap('<p class="pl-note">Nothing to show.</p>')

    biggest = max(value for _, value in items) or 1.0
    rows = []
    for label, value in items:
        is_cut = kept is not None and label not in kept
        width = max(value / biggest * 100.0, 0.0)
        rows.append(
            f'<div class="pl-bar{" is-cut" if is_cut else ""}">'
            f'<div class="pl-bar-label">{escape(label)}</div>'
            f'<div class="pl-bar-track">'
            f'<div class="pl-bar-fill" style="width:{width:.2f}%"></div></div>'
            f'<div class="pl-bar-value">{escape(value_format.format(value))}</div>'
            f"</div>"
        )

    heading = f'<p class="pl-title">{escape(title)}</p>' if title else ""
    return wrap(f'{heading}<div class="pl-bars">{"".join(rows)}</div>')


def heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    show_values: bool = False,
    title: str | None = None,
) -> str:
    """A sequential heatmap. Exact zeros are drawn as hatching, not as colour.

    That distinction matters on the attention slide: a masked cell is not "a very
    small weight", it is a cell the model is structurally forbidden to look at.
    """
    columns = len(col_labels)
    cells = ['<div class="pl-heat-label"></div>']
    for label in col_labels:
        cells.append(f'<div class="pl-heat-label is-col">{escape(label)}</div>')

    for row_index, row in enumerate(matrix):
        cells.append(
            f'<div class="pl-heat-label is-row">{escape(row_labels[row_index])}</div>'
        )
        for col_index, value in enumerate(row):
            tip = (
                f"{row_labels[row_index]} → {col_labels[col_index]}: "
                f"{'masked' if value == 0.0 else format(value, '.3f')}"
            )
            if value == 0.0:
                cells.append(
                    f'<div class="pl-heat-cell is-masked" title="{escape(tip)}"></div>'
                )
                continue
            alpha = _HEAT_MIN_ALPHA + (1.0 - _HEAT_MIN_ALPHA) * value
            ink = "#fff" if value >= _HEAT_TEXT_FLIP else "var(--text-primary)"
            text = f"{value:.2f}"[1:] if show_values and value >= 0.01 else ""
            # Bigger type for a bigger weight: the word being looked at hardest
            # should read as the biggest number, not just the darkest cell.
            font_size = _HEAT_FONT_MIN + (_HEAT_FONT_MAX - _HEAT_FONT_MIN) * value
            cells.append(
                f'<div class="pl-heat-cell" style="background:rgba(var(--heat-hue),'
                f'{alpha:.3f});color:{ink};font-size:{font_size:.2f}rem" '
                f'title="{escape(tip)}">{text}</div>'
            )

    heading = f'<p class="pl-title">{escape(title)}</p>' if title else ""
    grid = (
        f'<div class="pl-heat" style="grid-template-columns:auto repeat({columns},auto)">'
        f'{"".join(cells)}</div>'
    )
    return wrap(f'{heading}<div class="pl-scroll">{grid}</div>')


def _declutter(
    left: float, top: float, placed: list[tuple[float, float]], min_gap: float = 4.5
) -> float:
    """Nudge a point down until it clears whatever was already placed nearby.

    Points routinely land close together in the toy embedding space -- that's
    the point of the analogy demo -- so overlapping labels are common. The
    nudge is small and only kicks in on a real collision, so it stays close
    to the true coordinates while keeping the text readable.
    """
    while any(
        abs(left - other_left) < min_gap and abs(top - other_top) < min_gap
        for other_left, other_top in placed
    ):
        top += min_gap
    return top


def scatter(
    points: list[tuple[str, float, float, str]],
    legend: list[tuple[str, str]] | None = None,
    title: str | None = None,
) -> str:
    """A 2D scatter of labelled points.

    ``points`` is ``(label, x, y, role)`` where role is ""/"selected"/
    "neighbour"/"target". Every point carries its own text label, so identity
    never depends on colour alone.
    """
    if not points:
        return wrap('<p class="pl-note">Nothing to show.</p>')

    xs = [x for _, x, _, _ in points]
    ys = [y for _, _, y, _ in points]
    pad = 0.6
    x_min, x_max = min(xs) - pad, max(xs) + pad
    y_min, y_max = min(ys) - pad, max(ys) + pad
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0

    dots = []
    placed: list[tuple[float, float]] = []
    for label, x, y, role in points:
        left = (x - x_min) / x_span * 100.0
        # SVG/CSS y grows downward; flip so the maths matches the picture.
        top = (1.0 - (y - y_min) / y_span) * 100.0
        # The analogy target is designed to land exactly on its answer word --
        # that overlap *is* the point. A small nudge keeps both dots visible
        # without the usual label-crowding gap shoving it rows away.
        gap = 1.2 if role == "target" else 4.5
        top = _declutter(left, top, placed, min_gap=gap)
        placed.append((left, top))
        cls = f" is-{role}" if role else ""
        dots.append(
            f'<div class="pl-dot{cls}" style="left:{left:.2f}%;top:{top:.2f}%">'
            f"{escape(label)}</div>"
        )

    legend_html = ""
    if legend:
        entries = "".join(
            f'<span class="pl-legend-item">'
            f'<span class="pl-legend-swatch" style="background:{colour}"></span>'
            f"{escape(name)}</span>"
            for name, colour in legend
        )
        legend_html = f'<div class="pl-legend">{entries}</div>'

    heading = f'<p class="pl-title">{escape(title)}</p>' if title else ""
    return wrap(
        f'{heading}<div class="pl-scatter">{"".join(dots)}</div>{legend_html}'
    )


def table(
    headers: list[str],
    rows: list[list[str]],
    token_column: int | None = None,
    highlight_last: int = 0,
    max_height: str | None = None,
) -> str:
    """A compact table. ``token_column`` is rendered in monospace accent colour.

    ``max_height`` (any CSS length) caps the table and lets it scroll vertically
    instead of growing the slide, keeping the header row pinned.
    """
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = []
    for index, row in enumerate(rows):
        is_new = index >= len(rows) - highlight_last if highlight_last else False
        cells = "".join(
            f'<td class="is-token">{escape(cell)}</td>'
            if i == token_column
            else f"<td>{escape(cell)}</td>"
            for i, cell in enumerate(row)
        )
        body.append(f'<tr class="{"is-new" if is_new else ""}">{cells}</tr>')

    scroll_class = "pl-scroll is-tall" if max_height else "pl-scroll"
    style = f' style="max-height:{max_height}"' if max_height else ""

    return wrap(
        f'<div class="{scroll_class}"{style}><table class="pl-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
        f"</table></div>"
    )


def note(message: str) -> str:
    return wrap(f'<p class="pl-note">{message}</p>')


def mismatch(
    scored_kind: str,
    scored_prompt: str,
    scored_answer: str,
    scored_detail: str,
    chat_kind: str,
    chat_prompt: str,
    chat_answer: str,
    chat_detail: str,
    connector: str = "same model",
) -> str:
    """Two cards, side by side: what the eval scored vs. what the model actually said.

    Built for the one case a slide needs to show a metric being right and the
    model being wrong at the same time -- a paragraph of prose does not make
    that contrast land the way two cards next to each other do.
    """

    def card(kind: str, prompt: str, answer: str, detail: str, good: bool) -> str:
        cls = "is-good" if good else "is-bad"
        mark = "✓" if good else "✗"
        return (
            '<div class="pl-mismatch-card">'
            f'<div class="pl-mismatch-kind">{escape(kind)}</div>'
            f'<div class="pl-mismatch-prompt">{escape(prompt)}</div>'
            f'<div class="pl-mismatch-answer {cls}">{escape(answer)} '
            f'<span class="pl-mismatch-mark">{mark}</span></div>'
            f'<div class="pl-mismatch-detail">{escape(detail)}</div>'
            "</div>"
        )

    return wrap(
        '<div class="pl-mismatch">'
        + card(scored_kind, scored_prompt, scored_answer, scored_detail, True)
        + f'<div class="pl-mismatch-versus">{escape(connector)}</div>'
        + card(chat_kind, chat_prompt, chat_answer, chat_detail, False)
        + "</div>"
    )
