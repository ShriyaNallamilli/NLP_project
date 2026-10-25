"""
Baseline comparison and cross-validation.

An accuracy figure on its own says nothing: 90% is excellent on a balanced
6-class problem and terrible on one where a single class holds 90% of the rows.
This module measures the trivial baselines that establish the floor, then a
ladder of representations, so each modelling choice has to justify itself.

Scores are reported on the validation fold, never on test. Cross-validation
runs over train+validation combined and reports mean +/- standard deviation,
which is what makes a gap between two models believable rather than noise.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from scripts.datasets.manifest import REPORT_DIR
from scripts.datasets.splits import RANDOM_SEED, Split, fit_frame

CV_FOLDS = 5


def _logistic() -> LogisticRegression:
    """The classifier used by the shipped pipeline."""
    return LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced", n_jobs=-1)


# Character n-grams over long documents are quadratically expensive to build
# and cross-validate: on the resume corpus (~4k characters per document) this
# single rung cost more wall-clock time than the other nine combined, for a
# representation that is not expected to win on long prose. It is measured on
# short-text corpora only, and skipped with a recorded reason on long ones.
MAX_CHARS_FOR_CHAR_NGRAMS = 600


def baseline_specs(
    max_features: int, include_char_ngrams: bool = True
) -> dict[str, Pipeline | DummyClassifier]:
    """
    Build the ladder of models to compare.

    The order matters: each rung isolates one change from the rung above, so a
    score difference is attributable. Trivial floors first, then representation
    changes at a fixed classifier, then classifier changes at a fixed
    representation.

    Args:
        max_features: Vocabulary cap shared by every vectoriser, so vocabulary
            size is never the hidden variable behind a difference.
        include_char_ngrams: Include the character n-gram rung. Disabled for
            long-document corpora where it is prohibitively slow.

    Returns:
        Named unfitted estimators.
    """
    tfidf_word = dict(
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )
    specs: dict[str, Pipeline | DummyClassifier] = {
        # Floors: what you get for free.
        "majority_class": DummyClassifier(strategy="most_frequent"),
        "stratified_random": DummyClassifier(
            strategy="stratified", random_state=RANDOM_SEED
        ),
        # Representation ladder, classifier held fixed at logistic regression.
        "bow_unigram": Pipeline(
            [
                ("vec", CountVectorizer(lowercase=True, max_features=max_features, min_df=2)),
                ("clf", _logistic()),
            ]
        ),
        "bow_unigram_stopwords": Pipeline(
            [
                (
                    "vec",
                    CountVectorizer(
                        lowercase=True,
                        stop_words="english",
                        max_features=max_features,
                        min_df=2,
                    ),
                ),
                ("clf", _logistic()),
            ]
        ),
        "tfidf_unigram": Pipeline(
            [("vec", TfidfVectorizer(ngram_range=(1, 1), **tfidf_word)), ("clf", _logistic())]
        ),
        "tfidf_1_2gram": Pipeline(
            [("vec", TfidfVectorizer(ngram_range=(1, 2), **tfidf_word)), ("clf", _logistic())]
        ),
        "tfidf_1_3gram": Pipeline(
            [("vec", TfidfVectorizer(ngram_range=(1, 3), **tfidf_word)), ("clf", _logistic())]
        ),
        # Classifier ladder, representation held fixed at tfidf 1-2gram.
        "tfidf_1_2gram_naive_bayes": Pipeline(
            [("vec", TfidfVectorizer(ngram_range=(1, 2), **tfidf_word)), ("clf", MultinomialNB())]
        ),
        "tfidf_1_2gram_linear_svm": Pipeline(
            [
                ("vec", TfidfVectorizer(ngram_range=(1, 2), **tfidf_word)),
                ("clf", LinearSVC(class_weight="balanced")),
            ]
        ),
    }

    if include_char_ngrams:
        specs["tfidf_char_3_5gram"] = Pipeline(
            [
                (
                    "vec",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=max_features,
                        sublinear_tf=True,
                        min_df=2,
                    ),
                ),
                ("clf", _logistic()),
            ]
        )
    return specs


def run_baselines(
    split: Split, name: str, max_features: int = 50_000, cross_validate: bool = True
) -> dict:
    """
    Fit every baseline on train and score it on validation.

    Args:
        split: The corpus partition. Test is not touched.
        name: Corpus name, used for the report filename.
        max_features: Vocabulary cap shared across vectorisers.
        cross_validate: Also run stratified k-fold over train+validation.

    Returns:
        A report dict, also written to ``data/reports/<name>_baselines.json``.
    """
    mean_chars = float(split.train["text"].str.len().mean())
    include_char = mean_chars <= MAX_CHARS_FOR_CHAR_NGRAMS
    models = baseline_specs(
        max_features=max_features, include_char_ngrams=include_char
    )
    if not include_char:
        print(
            f"  (skipping char n-grams: mean document is {mean_chars:.0f} chars, "
            f"over the {MAX_CHARS_FOR_CHAR_NGRAMS} limit)"
        )

    combined = fit_frame(split, include_val=True)
    rows: list[dict] = []

    for position, (model_name, estimator) in enumerate(models.items(), start=1):
        started = datetime.now(timezone.utc)
        print(f"  [{position}/{len(models)}] {model_name} ...", end="", flush=True)

        estimator.fit(split.train["text"], split.train["label"])
        predicted = estimator.predict(split.val["text"])

        row = {
            "model": model_name,
            "val_accuracy": float(accuracy_score(split.val["label"], predicted)),
            "val_f1_macro": float(
                f1_score(split.val["label"], predicted, average="macro", zero_division=0)
            ),
        }

        if cross_validate:
            folds = StratifiedKFold(
                n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
            )
            scores = cross_val_score(
                estimator,
                combined["text"],
                combined["label"],
                cv=folds,
                scoring="accuracy",
                n_jobs=1,
            )
            row["cv_mean"] = float(np.mean(scores))
            row["cv_std"] = float(np.std(scores))
            row["cv_folds"] = CV_FOLDS

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        print(f" val {row['val_accuracy']:.1%} ({elapsed:.0f}s)", flush=True)
        rows.append(row)

    rows.sort(key=lambda item: item["val_accuracy"], reverse=True)
    label_counts = split.train["label"].value_counts()

    report = {
        "corpus": name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split.sizes,
        "classes": int(split.train["label"].nunique()),
        "majority_class_share": float(label_counts.iloc[0] / label_counts.sum()),
        "cross_validated_on": int(len(combined)),
        "mean_document_chars": round(mean_chars, 1),
        "char_ngrams_included": include_char,
        "char_ngrams_skipped_reason": (
            None
            if include_char
            else f"mean document {mean_chars:.0f} chars exceeds "
            f"{MAX_CHARS_FOR_CHAR_NGRAMS}; char n-grams are prohibitively slow "
            "to cross-validate on long documents"
        ),
        "results": rows,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / f"{name}_baselines.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(destination)
    return report


def format_table(report: dict) -> str:
    """
    Render a baseline report as a fixed-width table.

    Args:
        report: Output of :func:`run_baselines`.

    Returns:
        A printable table with the trivial floor called out.
    """
    lines = [
        f"corpus: {report['corpus']}  "
        f"classes: {report['classes']}  "
        f"majority-class share: {report['majority_class_share']:.1%}",
        f"split: train {report['split']['train']} / val {report['split']['val']} / "
        f"test {report['split']['test']} (test untouched)",
        "",
        f"{'model':<28}{'val acc':>9}{'val F1':>9}   {f'{CV_FOLDS}-fold CV (mean +/- sd)':>24}",
        "-" * 74,
    ]
    for row in report["results"]:
        cv = ""
        if "cv_mean" in row:
            cv = f"{row['cv_mean']:.3f} +/- {row['cv_std']:.3f}"
        lines.append(
            f"{row['model']:<28}"
            f"{row['val_accuracy']:>8.1%}"
            f"{row['val_f1_macro']:>9.3f}"
            f"   {cv:>24}"
        )
    return "\n".join(lines)


def lift_over_floor(report: dict) -> str:
    """
    State how much the best model beats the trivial baseline.

    Args:
        report: Output of :func:`run_baselines`.

    Returns:
        A one-line summary, the sentence a report needs next to an accuracy.
    """
    by_name = {row["model"]: row for row in report["results"]}
    floor = by_name.get("majority_class")
    best = report["results"][0]
    if not floor:
        return f"best: {best['model']} at {best['val_accuracy']:.1%}"
    delta = best["val_accuracy"] - floor["val_accuracy"]
    return (
        f"{best['model']} reaches {best['val_accuracy']:.1%}, "
        f"{delta:.1%} above the majority-class floor of "
        f"{floor['val_accuracy']:.1%}"
    )
