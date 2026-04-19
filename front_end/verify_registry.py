"""
Checks that BOOK_STOCKS in stock_registry.py exactly matches
the CSV stems in individual_book_train/.

Run from the repo root:
    python front_end/verify_registry.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from stock_registry import BOOK_STOCKS  # noqa: E402

FOLDER = Path(__file__).resolve().parents[1] / "individual_book_train"

actual = {p.stem for p in FOLDER.glob("*.csv")}
registered = set(BOOK_STOCKS)

missing = actual - registered
extra = registered - actual

if missing:
    print(f"IN FOLDER but missing from registry ({len(missing)}):")
    for s in sorted(missing, key=lambda x: int(x.split("_")[1])):
        print(f"  {s}")

if extra:
    print(f"IN REGISTRY but not in folder ({len(extra)}):")
    for s in sorted(extra, key=lambda x: int(x.split("_")[1])):
        print(f"  {s}")

if not missing and not extra:
    print(f"OK — registry matches folder exactly ({len(actual)} stocks)")
