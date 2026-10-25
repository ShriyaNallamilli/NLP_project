"""
Download the NLTK corpora the platform uses.

The app degrades to regex text processing when a corpus is missing, so this
step is not mandatory - but running it once up front means the first candidate
answer of the day is not the thing that pays for the download, and it is the
only way to prepare a host that has NLTK_AUTO_DOWNLOAD turned off.

Usage:
    python -m scripts.setup_nltk
    python -m scripts.setup_nltk --check     # report status, download nothing
"""
from __future__ import annotations

import argparse
import sys

try:
    import nltk
except ImportError:
    sys.exit(
        "nltk is not installed in this environment.\n"
        "Install it with: pip install -r requirements.txt"
    )

# Kept in step with app/utils/nlp.py RESOURCES: download id -> lookup path.
RESOURCES: dict[str, str] = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "vader_lexicon": "sentiment/vader_lexicon",
}


def is_present(path: str) -> bool:
    """
    Check whether a corpus is already installed.

    wordnet and vader_lexicon stay zipped after download, so they resolve
    only as "<path>.zip"; an unzipped install resolves only as "<path>".

    Args:
        path: nltk.data lookup path, without the .zip suffix.

    Returns:
        True when the corpus can be found locally.
    """
    for candidate in (path, f"{path}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
        except Exception:  # noqa: BLE001 - a broken data dir reads as missing
            return False
    return False


def main() -> int:
    """
    Download every required corpus and report the outcome.

    Returns:
        0 when everything is present, 1 when at least one corpus is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what is installed without downloading anything",
    )
    args = parser.parse_args()

    print(f"NLTK {nltk.__version__}")
    print(f"Search path: {nltk.data.path[0]}\n")

    missing: list[str] = []
    for name, path in RESOURCES.items():
        if is_present(path):
            print(f"  {name:38} already present")
            continue

        if args.check:
            print(f"  {name:38} MISSING")
            missing.append(name)
            continue

        print(f"  {name:38} downloading...", end=" ", flush=True)
        try:
            nltk.download(name, quiet=True)
        except Exception as exc:  # noqa: BLE001 - report, do not traceback
            print(f"FAILED ({exc})")
            missing.append(name)
            continue

        if is_present(path):
            print("ok")
        else:
            print("FAILED (not found after download)")
            missing.append(name)

    if missing:
        print(
            f"\n{len(missing)} corpus/corpora unavailable: {', '.join(missing)}\n"
            "The app still runs - it falls back to regex text processing and "
            "logs a warning when it does."
        )
        return 1

    print("\nAll NLTK corpora are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
