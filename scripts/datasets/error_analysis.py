"""
Error analysis for the trained classifiers.

A confusion matrix is a wall of numbers; what a reader needs is the handful of
facts inside it. This module extracts those: which class pairs the model
actually confuses, which classes it never predicts at all, whether recall
tracks training support, and concrete misclassified examples to read.

The aim is to answer *why* a number is what it is - the difference between
reporting 67.8% and explaining it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from scripts.datasets.manifest import REPORT_DIR

# A class predicted for fewer than this share of its true rows is "starved".
STARVED_RECALL = 0.35
TOP_CONFUSIONS = 12
EXAMPLES_PER_PAIR = 2


def confusion_pairs(
    truth: pd.Series, predicted: np.ndarray, labels: list[str], top: int = TOP_CONFUSIONS
) -> list[dict]:
    """
    Rank the most frequent (true, predicted) mistakes.

    Args:
        truth: True labels.
        predicted: Model predictions.
        labels: Label ordering for the matrix.
        top: How many pairs to return.

    Returns:
        Pairs sorted by count, each with the share of that class's rows lost.
    """
    matrix = confusion_matrix(truth, predicted, labels=labels)
    support = matrix.sum(axis=1)
    pairs: list[dict] = []

    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            if i == j or matrix[i, j] == 0:
                continue
            pairs.append(
                {
                    "true": true_label,
                    "predicted": predicted_label,
                    "count": int(matrix[i, j]),
                    "share_of_true_class": (
                        float(matrix[i, j] / support[i]) if support[i] else 0.0
                    ),
                }
            )

    pairs.sort(key=lambda pair: pair["count"], reverse=True)
    return pairs[:top]


def support_vs_recall(report: dict, train_counts: pd.Series) -> dict:
    """
    Test whether per-class recall tracks how much training data the class had.

    A strong positive correlation means the weak classes are weak because they
    are rare - a data problem. A weak correlation means something else is going
    on, such as genuine semantic overlap between classes.

    Args:
        report: sklearn ``classification_report`` output dict.
        train_counts: Rows per label in the training fold.

    Returns:
        The per-class table and the Pearson correlation between the two.
    """
    rows: list[dict] = []
    for label, metrics in report.items():
        if not isinstance(metrics, dict) or label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "label": label,
                "train_rows": int(train_counts.get(label, 0)),
                "recall": float(metrics.get("recall", 0.0)),
                "precision": float(metrics.get("precision", 0.0)),
                "f1": float(metrics.get("f1-score", 0.0)),
                "test_support": int(metrics.get("support", 0)),
            }
        )

    rows.sort(key=lambda row: row["recall"])
    correlation = 0.0
    if len(rows) > 2:
        supports = np.array([row["train_rows"] for row in rows], dtype=float)
        recalls = np.array([row["recall"] for row in rows], dtype=float)
        if supports.std() > 0 and recalls.std() > 0:
            correlation = float(np.corrcoef(supports, recalls)[0, 1])

    return {"per_class": rows, "support_recall_correlation": correlation}


def sample_errors(
    texts: pd.Series,
    truth: pd.Series,
    predicted: np.ndarray,
    pairs: list[dict],
    per_pair: int = EXAMPLES_PER_PAIR,
    excerpt: int = 240,
) -> list[dict]:
    """
    Pull concrete misclassified texts for the top confusion pairs.

    Args:
        texts: The input texts.
        truth: True labels.
        predicted: Model predictions.
        pairs: Confusion pairs to sample from.
        per_pair: Examples per pair.
        excerpt: Characters of text to keep.

    Returns:
        Example rows, for reading rather than for metrics.
    """
    frame = pd.DataFrame(
        {"text": texts.to_numpy(), "true": truth.to_numpy(), "predicted": predicted}
    )
    examples: list[dict] = []
    for pair in pairs[:6]:
        matching = frame[
            (frame["true"] == pair["true"]) & (frame["predicted"] == pair["predicted"])
        ].head(per_pair)
        for _, row in matching.iterrows():
            examples.append(
                {
                    "true": pair["true"],
                    "predicted": pair["predicted"],
                    "excerpt": str(row["text"])[:excerpt],
                }
            )
    return examples


def analyse(
    model_name: str,
    pipeline,
    frame: pd.DataFrame,
    train_counts: pd.Series,
    fold: str = "test",
) -> dict:
    """
    Run the full error analysis for one model.

    Args:
        model_name: Name used for the report file.
        pipeline: A fitted estimator.
        frame: The held-out fold to analyse.
        train_counts: Rows per label in the training fold.
        fold: Which fold this is, recorded in the report.

    Returns:
        The analysis dict, also written to
        ``data/reports/<model_name>_errors.json``.
    """
    predicted = pipeline.predict(frame["text"])
    truth = frame["label"]
    labels = sorted(set(truth) | set(predicted))

    per_class_report = classification_report(
        truth, predicted, output_dict=True, zero_division=0
    )
    pairs = confusion_pairs(truth, predicted, labels)
    support = support_vs_recall(per_class_report, train_counts)

    never_predicted = sorted(set(truth) - set(predicted))
    starved = [
        row["label"]
        for row in support["per_class"]
        if row["recall"] < STARVED_RECALL and row["label"] in set(truth)
    ]

    analysis = {
        "model": model_name,
        "fold": fold,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(frame)),
        "classes": len(labels),
        "accuracy": float((predicted == truth).mean()),
        "never_predicted": never_predicted,
        "starved_classes": starved,
        "top_confusions": pairs,
        "support_recall_correlation": support["support_recall_correlation"],
        "per_class": support["per_class"],
        "examples": sample_errors(frame["text"], truth, predicted, pairs),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / f"{model_name}_errors.json"
    destination.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    analysis["report_path"] = str(destination)
    return analysis


def format_analysis(analysis: dict) -> str:
    """
    Render an error analysis as readable prose and tables.

    Args:
        analysis: Output of :func:`analyse`.

    Returns:
        A printable summary suitable for pasting into a report.
    """
    lines = [
        f"{analysis['model']} on {analysis['fold']}: "
        f"{analysis['accuracy']:.1%} over {analysis['rows']} rows, "
        f"{analysis['classes']} classes",
        "",
        "Worst classes (lowest recall first):",
        f"  {'class':<28}{'train':>7}{'test':>6}{'recall':>9}{'prec':>8}",
    ]
    for row in analysis["per_class"][:8]:
        lines.append(
            f"  {row['label'][:26]:<28}"
            f"{row['train_rows']:>7}{row['test_support']:>6}"
            f"{row['recall']:>9.2f}{row['precision']:>8.2f}"
        )

    lines += ["", "Most frequent confusions:"]
    for pair in analysis["top_confusions"][:8]:
        lines.append(
            f"  {pair['true'][:24]:<26} -> {pair['predicted'][:24]:<26}"
            f"{pair['count']:>4}  ({pair['share_of_true_class']:.0%} of class)"
        )

    correlation = analysis["support_recall_correlation"]
    lines += ["", "Interpretation:"]
    if analysis["never_predicted"]:
        lines.append(
            f"  Never predicted at all: {', '.join(analysis['never_predicted'])}."
        )
    if analysis["starved_classes"]:
        lines.append(
            f"  Recall below {STARVED_RECALL:.0%}: "
            f"{', '.join(analysis['starved_classes'][:6])}."
        )
    strength = (
        "strong" if abs(correlation) > 0.6
        else "moderate" if abs(correlation) > 0.3
        else "weak"
    )
    lines.append(
        f"  Correlation between training rows and recall: {correlation:+.2f} "
        f"({strength})."
    )
    if abs(correlation) > 0.3:
        lines.append(
            "  Rare classes underperform: more data for them is the first fix."
        )
    else:
        lines.append(
            "  Class size does not explain the weak classes; the top confusion "
            "pairs above point at genuine overlap between categories instead."
        )
    return "\n".join(lines)
