"""
NLTK gateway for the text processing shared across the platform.

Every NLTK entry point here is optional. A corpus can be missing on a fresh
checkout, a locked-down host can refuse the download, and neither may take an
interview down - so each function falls back to the regex behaviour the
project used before NLTK arrived, and says so in the log exactly once.

Run ``python -m scripts.setup_nltk`` to fetch the corpora up front. Set
``NLTK_AUTO_DOWNLOAD=0`` to forbid downloads at runtime, which is the right
setting for an offline or production host that provisions them at build time.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Optional

try:
    import nltk

    NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover - nltk is a declared dependency
    nltk = None
    NLTK_AVAILABLE = False

_LOGGER = logging.getLogger(__name__)

# Each corpus is looked up by the path nltk.data.find() expects, which is not
# always the download id: the tagger and punkt tables differ between the two.
# Some corpora are left zipped by the downloader and only resolve as
# "<path>.zip", so _find() tries both spellings.
RESOURCES: dict[str, str] = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
    "vader_lexicon": "sentiment/vader_lexicon",
}

_WORD_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9'+.#-]*")

# NLTK splits "don't" into "do" + "n't" and "it's" into "it" + "'s". Counting
# the clitic as its own word inflates every word count and therefore the
# words-per-minute figure, so the pieces are dropped and the contraction
# counts as the one word a person would call it.
_CLITICS = frozenset({"n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "'t"})

# Used when the stopwords corpus is unreachable. Short, and only has to be good
# enough to stop the most common filler from being treated as content.
_FALLBACK_STOPWORDS = frozenset(
    """a an and are as at be been by for from had has have he her his i in is it
    its me my of on or our she that the their them they this to was we were with
    you your""".split()
)

_lock = threading.Lock()
_resource_state: dict[str, bool] = {}
_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """
    Log a degradation warning the first time it happens.

    Without the guard this fires on every answer of every interview, which
    buries the rest of the log.

    Args:
        key: Stable identifier for the warning.
        message: Text to log.
    """
    if key in _warned:
        return
    _warned.add(key)
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            current_app.logger.warning(message)
            return
    except Exception:  # pragma: no cover - flask always importable here
        pass
    _LOGGER.warning(message)


def _find(path: str) -> bool:
    """
    Look a corpus up under both its plain and zipped path.

    wordnet and vader_lexicon stay zipped after download and resolve only as
    "corpora/wordnet.zip"; an unzipped install resolves only as
    "corpora/wordnet". Either one counts as installed.

    Args:
        path: nltk.data lookup path, without the .zip suffix.

    Returns:
        True when the corpus is present in some form.
    """
    for candidate in (path, f"{path}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def _auto_download_allowed() -> bool:
    """Return True unless NLTK_AUTO_DOWNLOAD explicitly disables downloading."""
    return os.getenv("NLTK_AUTO_DOWNLOAD", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def ensure_resource(name: str) -> bool:
    """
    Make one NLTK corpus available, downloading it once if permitted.

    The result is cached for the process: a host with no network should pay
    the timeout once, not on every request.

    Args:
        name: Key from :data:`RESOURCES`.

    Returns:
        True when the corpus can be loaded.
    """
    if not NLTK_AVAILABLE:
        return False

    cached = _resource_state.get(name)
    if cached is not None:
        return cached

    with _lock:
        cached = _resource_state.get(name)
        if cached is not None:
            return cached

        path = RESOURCES.get(name, name)
        available = False
        try:
            available = _find(path)
            if not available and _auto_download_allowed():
                nltk.download(name, quiet=True)
                available = _find(path)
                if not available:
                    _warn_once(
                        f"download:{name}",
                        f"NLTK corpus '{name}' is still missing after a "
                        f"download attempt. Falling back to regex text "
                        f"processing. Run 'python -m scripts.setup_nltk'.",
                    )
            elif not available:
                _warn_once(
                    f"disabled:{name}",
                    f"NLTK corpus '{name}' is missing and NLTK_AUTO_DOWNLOAD is "
                    f"off. Falling back to regex text processing. Run "
                    f"'python -m scripts.setup_nltk' to install it.",
                )
        except Exception as exc:  # noqa: BLE001 - never fail a request
            _warn_once(
                f"lookup:{name}",
                f"NLTK corpus '{name}' could not be loaded ({exc}). "
                f"Falling back to regex text processing. Run "
                f"'python -m scripts.setup_nltk' to install it.",
            )

        _resource_state[name] = available
        return available


def word_tokens(text: str) -> list[str]:
    """
    Split text into lowercased word tokens.

    NLTK is preferred because ``str.split()`` leaves punctuation attached, so
    "um," and "um" counted as different words and every sentence-final word
    was miscounted.

    Args:
        text: Raw text.

    Returns:
        Lowercased tokens; punctuation-only tokens and contraction clitics
        are removed.
    """
    if not text:
        return []

    if ensure_resource("punkt") and ensure_resource("punkt_tab"):
        try:
            from nltk.tokenize import word_tokenize

            return [
                lowered
                for lowered in (token.lower() for token in word_tokenize(text))
                if lowered not in _CLITICS
                and any(char.isalnum() for char in lowered)
            ]
        except Exception as exc:  # noqa: BLE001
            _warn_once(
                "tokenize",
                f"NLTK word tokenizer failed ({exc}). Falling back to regex "
                f"tokenizing.",
            )

    # The pattern has to allow "." and "-" mid-token so "node.js" and
    # "co-located" survive, which means a sentence-final "world." matches
    # with its full stop attached. Strip those trailing characters back off.
    # "+" and "#" are left alone: "c++" and "c#" end in them legitimately.
    tokens = []
    for match in _WORD_RE.finditer(text):
        token = match.group(0).lower().rstrip(".'-")
        if token:
            tokens.append(token)
    return tokens


def stopword_set() -> frozenset[str]:
    """
    Return English stopwords, or a small built-in list when the corpus is gone.

    Returns:
        Lowercased stopwords.
    """
    if ensure_resource("stopwords"):
        try:
            from nltk.corpus import stopwords

            return frozenset(stopwords.words("english"))
        except Exception as exc:  # noqa: BLE001
            _warn_once(
                "stopwords",
                f"NLTK stopwords failed to load ({exc}). Falling back to the "
                f"built-in stopword list.",
            )
    return _FALLBACK_STOPWORDS


def content_tokens(text: str) -> list[str]:
    """
    Tokenize and drop stopwords.

    Args:
        text: Raw text.

    Returns:
        Content-bearing lowercased tokens.
    """
    stops = stopword_set()
    return [token for token in word_tokens(text) if token not in stops]


def normalize_tokens(tokens: list[str]) -> set[str]:
    """
    Reduce tokens to a comparable base form.

    WordNet lemmatizing is tried first, then the Porter stemmer, which ships
    inside NLTK and needs no corpus. Both let an exact-match word list catch
    inflections - "cheating" against "cheat" - that plain equality misses.

    Args:
        tokens: Lowercased tokens.

    Returns:
        Set of base forms, always including the tokens themselves.
    """
    forms = set(tokens)
    if not tokens or not NLTK_AVAILABLE:
        return forms

    if ensure_resource("wordnet"):
        try:
            from nltk.stem import WordNetLemmatizer

            lemmatizer = WordNetLemmatizer()
            for token in tokens:
                forms.add(lemmatizer.lemmatize(token))
                forms.add(lemmatizer.lemmatize(token, pos="v"))
        except Exception as exc:  # noqa: BLE001
            _warn_once(
                "lemmatize",
                f"NLTK lemmatizer failed ({exc}). Falling back to stemming.",
            )

    try:
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        forms.update(stemmer.stem(token) for token in tokens)
    except Exception as exc:  # noqa: BLE001
        _warn_once("stem", f"NLTK stemmer failed ({exc}). Using raw tokens only.")

    return forms


def sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Args:
        text: Raw text.

    Returns:
        Sentence strings, blank ones removed.
    """
    if not text:
        return []

    if ensure_resource("punkt") and ensure_resource("punkt_tab"):
        try:
            from nltk.tokenize import sent_tokenize

            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception as exc:  # noqa: BLE001
            _warn_once(
                "sent_tokenize",
                f"NLTK sentence tokenizer failed ({exc}). Falling back to regex "
                f"sentence splitting.",
            )

    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def pos_tags(tokens: list[str]) -> Optional[list[tuple[str, str]]]:
    """
    Tag tokens with parts of speech.

    Args:
        tokens: Tokens to tag.

    Returns:
        (token, tag) pairs, or None when the tagger is unavailable - there is
        no sensible regex substitute for tagging, so callers skip the step.
    """
    if not tokens or not ensure_resource("averaged_perceptron_tagger_eng"):
        return None
    try:
        from nltk import pos_tag

        return pos_tag(tokens)
    except Exception as exc:  # noqa: BLE001
        _warn_once(
            "pos_tag",
            f"NLTK POS tagger failed ({exc}). Skipping part-of-speech analysis.",
        )
        return None


def status() -> dict[str, bool]:
    """
    Report which corpora are currently usable.

    Returns:
        Mapping of resource name to availability.
    """
    return {name: ensure_resource(name) for name in RESOURCES}
