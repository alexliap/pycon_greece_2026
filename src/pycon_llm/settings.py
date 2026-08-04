from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SLIDES_PATH = PROJECT_ROOT / "slides"
STYLES_PATH = PROJECT_ROOT / "styles"
ASSETS_PATH = PROJECT_ROOT / "assets"

# Committed tiktoken BPE ranks. Pointing tiktoken here means the deck never
# needs the network -- which matters when you are on a conference stage.
TIKTOKEN_CACHE_DIR = ASSETS_PATH / "tiktoken_cache"
