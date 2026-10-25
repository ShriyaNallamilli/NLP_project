"""
Evaluate the trained classifiers on a held-out split.

Training already reports headline metrics; this module exists for the checks
that matter once a model is wired into the app: a confusion matrix you can
actually read, the per-class recall that tells you which emotion the model
quietly never predicts, and the mapping from model probabilities onto the four
float columns EmotionAnalysis stores.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from scripts.datasets.loaders import load_resumes, load_text_emotions
from scripts.datasets.manifest import REPORT_DIR
from scripts.datasets.splits import make_split
from scripts.datasets.train import (
    EMOTION_MODEL_NAME,
    EMOTION_TO_APP,
    RESUME_MODEL_NAME,
    RESUME_TRAIN_SOURCES,
    load_model,
)

# EmotionAnalysis stores these four as separate floats.
APP_EMOTIONS = ("Happy", "Neutral", "Sad", "Nervous")


def evaluate_model(name: str, frame: pd.DataFrame) -> dict:
    """
    Score a trained model against the shared test fold.

    The split comes from :mod:`scripts.datasets.splits`, so this reads exactly
    the rows training held out - no fraction to keep in sync by hand.

    Args:
        name: Model base name.
        frame: The same corpus the model was trained on.

    Returns:
        A report dict, also written to ``data/reports/<name>_eval.json``.
    """
    pipeline = load_model(name)
    split = make_split(frame)
    x_test, y_test = split.test["text"], split.test["label"]
    predictions = pipeline.predict(x_test)
    labels = sorted(set(y_test) | set(predictions))

    matrix = confusion_matrix(y_test, predictions, labels=labels)
    per_class = classification_report(
        y_test, predictions, output_dict=True, zero_division=0
    )

    never_predicted = [
        label
        for label in labels
        if label in set(y_test) and label not in set(predictions)
    ]

    report = {
        "model": name,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows_evaluated": int(len(x_test)),
        "accuracy": float((predictions == y_test).mean()),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "never_predicted": never_predicted,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / f"{name}_eval.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(destination)
    return report


def evaluate_resume_classifier() -> dict:
    """Evaluate the resume category model against the corpus it was trained on."""
    return evaluate_model(
        RESUME_MODEL_NAME, load_resumes(sources=RESUME_TRAIN_SOURCES)
    )


def evaluate_emotion_classifier() -> dict:
    """Evaluate the text emotion model."""
    frame = load_text_emotions()
    frame["label"] = frame["label"].str.lower().str.strip()
    return evaluate_model(EMOTION_MODEL_NAME, frame)


def emotion_scores(text: str) -> dict[str, float]:
    """
    Convert one answer transcript into the four EmotionAnalysis scores.

    This is the bridge the app would call: corpus emotion labels are folded
    into the app's four buckets via ``EMOTION_TO_APP`` and the per-class
    probabilities are summed, so the result is directly assignable to
    ``happy_score`` / ``neutral_score`` / ``sad_score`` / ``nervous_score``.

    Args:
        text: The answer transcript.

    Returns:
        A dict with one float per app emotion plus ``dominant_emotion``.
        Scores are on a 0-100 scale to match the existing columns.
    """
    pipeline = load_model(EMOTION_MODEL_NAME)
    probabilities = pipeline.predict_proba([text])[0]
    classes = pipeline.named_steps["clf"].classes_

    buckets = {emotion: 0.0 for emotion in APP_EMOTIONS}
    for label, probability in zip(classes, probabilities):
        bucket = EMOTION_TO_APP.get(str(label).lower(), "Neutral")
        buckets[bucket] += float(probability)

    scores = {
        f"{emotion.lower()}_score": round(value * 100, 2)
        for emotion, value in buckets.items()
    }
    scores["dominant_emotion"] = max(buckets, key=buckets.get)
    return scores


def summarise(reports: list[dict]) -> str:
    """
    Render evaluation reports as a short text table for the terminal.

    Args:
        reports: Report dicts from :func:`evaluate_model`.

    Returns:
        A printable summary.
    """
    lines = [f"{'model':<20}{'rows':>8}{'accuracy':>11}  notes"]
    lines.append("-" * 62)
    for report in reports:
        note = ""
        if report.get("never_predicted"):
            note = "never predicts: " + ", ".join(report["never_predicted"])
        lines.append(
            f"{report['model']:<20}"
            f"{report['rows_evaluated']:>8}"
            f"{report['accuracy']:>10.1%}  {note}"
        )
    return "\n".join(lines)


def load_report(name: str) -> dict | None:
    """
    Read a previously written evaluation report.

    Args:
        name: Model base name.

    Returns:
        The parsed report, or ``None`` if it has not been generated.
    """
    path = Path(REPORT_DIR) / f"{name}_eval.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
