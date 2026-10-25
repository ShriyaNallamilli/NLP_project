"""
Neural sentence embeddings compared against the TF-IDF baseline.

This answers the obvious question a TF-IDF-only project invites: would a
pretrained transformer do better? Rather than fine-tuning - which is not
practical on CPU for a corpus this size - it uses a frozen sentence encoder as
a feature extractor and fits the same logistic regression on top. That isolates
the representation: identical classifier, identical split, identical metric, so
any difference is attributable to the features.

The encoder truncates at 256 word pieces, so ~83% of the average resume is
never seen; the report records that per corpus.

The measured outcome is the opposite of the obvious guess, in both directions.
On resumes the embeddings win (73.4% vs 68.3%) *despite* that truncation - the
opening of a resume is apparently enough to place it. On the emotion corpus,
where nothing is truncated, they lose badly (64.1% vs 91.1%): those sentences
share a near-identical frame ("i feel ..."), so they crowd into one region of
embedding space, while the single emotion word that carries the label is
exactly what a sparse bag-of-words picks out. Neither representation is simply
better - it depends on whether the label lives in the wording or the meaning.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from scripts.datasets.manifest import MODEL_DIR, REPORT_DIR
from scripts.datasets.splits import Split, make_split

# Small, fast, CPU-friendly, and the usual default for sentence embeddings.
ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_TOKENS = 256
EMBED_BATCH = 64

CACHE_DIR = MODEL_DIR / "embeddings"


class EncoderUnavailable(RuntimeError):
    """Raised when sentence-transformers or the encoder weights are missing."""


def _load_encoder():
    """
    Load the sentence encoder.

    Returns:
        A ``SentenceTransformer`` instance.

    Raises:
        EncoderUnavailable: If the package or weights cannot be loaded.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EncoderUnavailable(
            "sentence-transformers is not installed. Run:\n"
            "  python -m pip install sentence-transformers torch"
        ) from exc

    try:
        encoder = SentenceTransformer(ENCODER_NAME)
    except Exception as exc:  # noqa: BLE001 - network or cache failure
        raise EncoderUnavailable(
            f"Could not load {ENCODER_NAME}: {exc}. The weights download on "
            "first use and need network access."
        ) from exc

    encoder.max_seq_length = MAX_SEQ_TOKENS
    return encoder


def embed_texts(texts: pd.Series, cache_key: str, encoder=None) -> np.ndarray:
    """
    Encode texts to dense vectors, caching the result on disk.

    Encoding is the slow step and is fully deterministic, so it is cached by
    corpus and row count rather than recomputed per experiment.

    Args:
        texts: Texts to encode.
        cache_key: Identifier for the cache file.
        encoder: Optional preloaded encoder.

    Returns:
        Array of shape (len(texts), embedding_dim).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}_{len(texts)}.npy"
    if cache_path.is_file():
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            return cached

    model = encoder or _load_encoder()
    vectors = model.encode(
        texts.tolist(),
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    np.save(cache_path, vectors)
    return vectors


def compare(
    frame: pd.DataFrame,
    name: str,
    tfidf_val_accuracy: float | None = None,
    tfidf_name: str | None = None,
) -> dict:
    """
    Fit a classifier on frozen embeddings and report validation scores.

    Args:
        frame: Canonical frame with ``text`` and ``label``.
        name: Corpus name, used for cache and report filenames.
        tfidf_val_accuracy: The best sparse-baseline validation accuracy to
            compare against, so the report states a delta against a real
            competitor rather than a bare number.
        tfidf_name: Name of that baseline, recorded so the comparison is
            reproducible and obviously not cherry-picked.

    Returns:
        A report dict, also written to
        ``data/reports/<name>_embeddings.json``.

    Raises:
        EncoderUnavailable: If the encoder cannot be loaded.
    """
    split: Split = make_split(frame)
    encoder = _load_encoder()

    train_vectors = embed_texts(split.train["text"], f"{name}_train", encoder)
    val_vectors = embed_texts(split.val["text"], f"{name}_val", encoder)

    classifier = LogisticRegression(
        max_iter=2000, C=4.0, class_weight="balanced", n_jobs=-1
    )
    classifier.fit(train_vectors, split.train["label"])
    predicted = classifier.predict(val_vectors)

    accuracy = float(accuracy_score(split.val["label"], predicted))
    mean_chars = float(frame["text"].str.len().mean())

    report = {
        "corpus": name,
        "encoder": ENCODER_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding_dim": int(train_vectors.shape[1]),
        "max_seq_tokens": MAX_SEQ_TOKENS,
        "split": split.sizes,
        "val_accuracy": accuracy,
        "val_f1_macro": float(
            f1_score(split.val["label"], predicted, average="macro", zero_division=0)
        ),
        "mean_text_chars": round(mean_chars, 1),
        # ~4 characters per token is the usual English rule of thumb.
        "estimated_truncation": round(
            max(0.0, 1.0 - (MAX_SEQ_TOKENS * 4) / mean_chars), 3
        )
        if mean_chars
        else 0.0,
    }

    if tfidf_val_accuracy is not None:
        report["tfidf_val_accuracy"] = float(tfidf_val_accuracy)
        report["tfidf_model"] = tfidf_name or "tfidf"
        report["delta_vs_tfidf"] = round(accuracy - tfidf_val_accuracy, 4)
        report["verdict"] = (
            "embeddings win" if accuracy > tfidf_val_accuracy else "tfidf wins"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / f"{name}_embeddings.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(destination)
    return report


def format_report(report: dict) -> str:
    """
    Render an embedding comparison for the terminal.

    Args:
        report: Output of :func:`compare`.

    Returns:
        A printable summary.
    """
    lines = [
        f"corpus: {report['corpus']}",
        f"encoder: {report['encoder']} ({report['embedding_dim']}-dim, "
        f"{report['max_seq_tokens']} tokens max)",
        f"mean text length: {report['mean_text_chars']:.0f} chars "
        f"(~{report['estimated_truncation']:.0%} truncated)",
        "",
        f"  embeddings + logistic regression : {report['val_accuracy']:.1%} val "
        f"(macro-F1 {report['val_f1_macro']:.3f})",
    ]
    if "tfidf_val_accuracy" in report:
        lines.append(
            f"  best sparse baseline             : "
            f"{report['tfidf_val_accuracy']:.1%} val "
            f"({report.get('tfidf_model', 'tfidf')})"
        )
        delta = report["delta_vs_tfidf"]
        lines.append("")
        lines.append(
            f"  {report['verdict']} by {abs(delta):.1%} on the validation fold"
        )
    return "\n".join(lines)
