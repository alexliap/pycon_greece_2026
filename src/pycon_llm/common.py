"""Shared helpers for the deck."""

from .settings import SLIDES_PATH


def read_slide(slide: str) -> str:
    """Return the markdown source of ``slides/<slide>.md``."""
    path = SLIDES_PATH / f"{slide}.md"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in SLIDES_PATH.glob("*.md")))
        raise FileNotFoundError(
            f"No slide named {slide!r} at {path}. Available slides: {available}"
        )
    return path.read_text(encoding="utf-8")
