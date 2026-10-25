"""
Train TF-IDF text classifiers on the Kaggle corpora.

Two models are produced, both saved to ``data/models`` as joblib pipelines:

    resume_category.joblib  resume text -> job category
    text_emotion.joblib     answer sentence -> emotion

Each task gets the classifier its validation fold argues for: a linear SVM for
resume categorisation, where it is measurably better, and logistic regression
for emotion, where the SVM's edge is inside fold noise and the app needs
calibrated per-class probabilities anyway - EmotionAnalysis stores happy /
neutral / sad / nervous as separate floats, not a single hard label. See
:func:`build_pipeline` for the numbers behind that choice.

This module also mines a skill vocabulary from the resume corpus, which is what
widens ResumeParserService.COMMON_SKILLS from its 58 hand-written entries.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from scripts.datasets.loaders import (
    load_resumes,
    load_text_emotions,
    seed_skill_list,
    write_processed,
)
from scripts.datasets.manifest import MODEL_DIR, PROCESSED_DIR, REPORT_DIR
from scripts.datasets.splits import fit_frame, make_split

RESUME_MODEL_NAME = "resume_category"
EMOTION_MODEL_NAME = "text_emotion"
SKILL_VOCAB_FILE = "skills_vocab.json"

# Only resumes_large is fitted: see train_resume_classifier for why the two
# resume corpora must not be concatenated.
RESUME_TRAIN_SOURCES = ("resumes_large",)

# The emotion corpora use six labels; EmotionAnalysis has four columns.
# This mapping is an approximation, recorded here so it stays reviewable
# rather than being buried in the scoring code.
EMOTION_TO_APP: dict[str, str] = {
    "joy": "Happy",
    "love": "Happy",
    "surprise": "Neutral",
    "neutral": "Neutral",
    "sadness": "Sad",
    "sad": "Sad",
    "anger": "Nervous",
    "fear": "Nervous",
}

# Terms that dominate resume TF-IDF but are not skills. Ranking by
# discriminativeness (see mine_skill_vocabulary) removes most of these on its
# own; this list catches the stragglers that are genuinely category-skewed but
# still not skills, such as "staff" in a management corpus.
_SKILL_STOPWORDS = {
    "company", "name", "description", "responsibilities", "details", "month",
    "months", "year", "years", "project", "projects", "team", "work", "working",
    "experience", "education", "college", "university", "india", "exprience",
    "less", "state", "city", "role", "client", "b tech", "e mail", "skills",
    "new", "professional", "current", "training", "service", "services",
    "business", "customer", "including", "manager", "office", "high", "staff",
    "support", "time", "managed", "management", "development", "information",
    "knowledge", "ability", "strong", "excellent", "good", "various", "well",
    "used", "using", "use", "based", "provide", "provided", "ensure", "ensured",
    "responsible", "duties", "job", "position", "level", "department", "reports",
    "daily", "weekly", "monthly", "quality", "process", "processes", "system",
    "systems", "data", "customers", "clients", "products", "product", "sales",
    "school", "degree", "bachelor", "master", "certified", "certification",
    "summary", "highlights", "accomplishments", "activities", "present",
}

# A skill term is at least 3 characters so single tokens like "go" and "r"
# cannot enter the vocabulary, where substring matching would misfire badly.
_SKILL_TOKEN = re.compile(r"^[a-z][a-z0-9 .+#/-]{2,28}$")

# How much more concentrated in its category a term must be, versus the corpus
# as a whole, before it counts as category-distinctive vocabulary.
_MIN_DISCRIMINATIVENESS = 2.0


def build_pipeline(
    max_features: int = 50_000, ngram_max: int = 2, classifier: str = "logistic"
) -> Pipeline:
    """
    Build the shared TF-IDF pipeline.

    The classifier is selectable because the two tasks want different ones, and
    the validation folds say so (see ``scripts/datasets/baselines.py``):

    - Resume categorisation uses a linear SVM. Its 5-fold CV accuracy is
      0.672 +/- 0.008 against 0.642 +/- 0.018 for logistic regression - a gap
      several times the fold-to-fold spread, so it is a real improvement.
    - Emotion uses logistic regression. The SVM's edge there is 0.907 vs 0.904,
      inside one standard deviation and not meaningful, while logistic
      regression supplies the calibrated ``predict_proba`` that
      ``TextEmotionService`` needs to fill four separate score columns. An SVM
      would force an uncalibrated decision-function hack for no measured gain.

    Args:
        max_features: Vocabulary cap for the vectoriser.
        ngram_max: Largest n-gram to extract.
        classifier: ``"logistic"`` or ``"linear_svm"``.

    Returns:
        An unfitted scikit-learn pipeline.

    Raises:
        ValueError: On an unknown classifier name.
    """
    if classifier == "logistic":
        estimator = LogisticRegression(
            max_iter=1000, C=4.0, class_weight="balanced", n_jobs=-1
        )
    elif classifier == "linear_svm":
        estimator = LinearSVC(class_weight="balanced")
    else:
        raise ValueError(
            f"Unknown classifier '{classifier}'; expected 'logistic' or 'linear_svm'."
        )

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, ngram_max),
                    max_features=max_features,
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            ("clf", estimator),
        ]
    )


def train_text_classifier(
    frame: pd.DataFrame,
    name: str,
    max_features: int = 50_000,
    ngram_max: int = 2,
    refit_on_val: bool = True,
    classifier: str = "logistic",
) -> dict:
    """
    Fit, score and persist a text classifier using the shared three-way split.

    Validation and test are reported separately. Validation is the number to
    compare models on; test is reported once, at the end, and must not drive
    any decision - that is what keeps it an honest estimate.

    Args:
        frame: Canonical frame with ``text`` and ``label`` columns.
        name: Base name for the model and metrics files.
        max_features: Vectoriser vocabulary cap.
        ngram_max: Largest n-gram to extract.
        refit_on_val: After scoring, refit on train+validation so the shipped
            model uses all non-test data. Reported metrics come from the
            train-only fit, so they stay comparable to the baselines.

    Returns:
        A metrics dict, also written to ``data/reports/<name>_metrics.json``.

    Raises:
        ValueError: If fewer than two classes survive filtering.
    """
    split = make_split(frame)

    pipeline = build_pipeline(
        max_features=max_features, ngram_max=ngram_max, classifier=classifier
    )
    pipeline.fit(split.train["text"], split.train["label"])

    val_predictions = pipeline.predict(split.val["text"])
    test_predictions = pipeline.predict(split.test["text"])

    metrics = {
        "model": name,
        "classifier": classifier,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split.sizes,
        "dropped_rare_labels": list(split.dropped_labels),
        "rows_total": int(sum(split.sizes.values())),
        "classes": sorted(split.train["label"].unique().tolist()),
        "val_accuracy": float((val_predictions == split.val["label"]).mean()),
        "val_f1_macro": float(
            f1_score(split.val["label"], val_predictions, average="macro", zero_division=0)
        ),
        "accuracy": float((test_predictions == split.test["label"]).mean()),
        "f1_macro": float(
            f1_score(split.test["label"], test_predictions, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(
                split.test["label"], test_predictions, average="weighted", zero_division=0
            )
        ),
        "per_class": classification_report(
            split.test["label"], test_predictions, output_dict=True, zero_division=0
        ),
        "sources": sorted(frame["source"].unique().tolist())
        if "source" in frame
        else [],
    }

    if refit_on_val:
        combined = fit_frame(split, include_val=True)
        pipeline = build_pipeline(
            max_features=max_features, ngram_max=ngram_max, classifier=classifier
        )
        pipeline.fit(combined["text"], combined["label"])
        metrics["refit_on"] = "train+val"
        metrics["refit_rows"] = int(len(combined))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(pipeline, model_path)
    report_path = REPORT_DIR / f"{name}_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics["model_path"] = str(model_path)
    metrics["report_path"] = str(report_path)
    return metrics


def train_resume_classifier(
    max_rows: int | None = None, sources: tuple[str, ...] = RESUME_TRAIN_SOURCES
) -> dict:
    """
    Train the resume text to job-category classifier.

    Trains on one corpus at a time by design. The two resume datasets label at
    different granularities - industry versus IT role - so concatenating them
    yields 45 overlapping classes and a model that is worse at both jobs.

    Args:
        max_rows: Optional per-dataset row cap.
        sources: Dataset keys to train on. Defaults to ``resumes_large``, the
            only corpus with enough unique rows per class to be worth fitting;
            ``resumes_classic`` dedupes down to ~166 rows across 25 classes.

    Returns:
        The metrics dict from :func:`train_text_classifier`.
    """
    frame = load_resumes(max_rows=max_rows, sources=sources)
    write_processed(frame, "resumes")
    return train_text_classifier(
        frame, RESUME_MODEL_NAME, max_features=60_000, classifier="linear_svm"
    )


def train_emotion_classifier(max_rows: int | None = None) -> dict:
    """
    Train the sentence to emotion classifier used on answer transcripts.

    Args:
        max_rows: Optional per-dataset row cap.

    Returns:
        The metrics dict, extended with the app-emotion mapping used.
    """
    frame = load_text_emotions(max_rows=max_rows)
    frame["label"] = frame["label"].str.lower().str.strip()
    write_processed(frame, "text_emotions")
    # Logistic regression, not the marginally better SVM: TextEmotionService
    # needs predict_proba to populate four separate score columns.
    metrics = train_text_classifier(
        frame, EMOTION_MODEL_NAME, max_features=40_000, classifier="logistic"
    )
    metrics["app_emotion_mapping"] = {
        label: EMOTION_TO_APP.get(label, "Neutral") for label in metrics["classes"]
    }
    return metrics


def mine_skill_vocabulary(top_per_category: int = 60, min_documents: int = 5) -> dict:
    """
    Mine candidate skill terms from the resume corpus.

    Terms are ranked per job category by TF-IDF so that category-distinctive
    vocabulary rises above generic resume boilerplate. Everything already in
    ``COMMON_SKILLS`` is marked ``confirmed``; newly mined terms are marked
    ``candidate`` and are meant to be reviewed before they are trusted - term
    mining is noisy and will surface some non-skills.

    Args:
        top_per_category: Candidate terms to keep per job category.
        min_documents: Minimum document frequency for a term to be kept.

    Returns:
        The vocabulary dict, also written to ``data/processed/skills_vocab.json``.
    """
    frame = load_resumes()
    seed = seed_skill_list()
    seed_lower = {item.lower() for item in seed}

    vectoriser = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 3),
        max_features=80_000,
        min_df=min_documents,
        sublinear_tf=True,
    )
    matrix = vectoriser.fit_transform(frame["text"])
    terms = vectoriser.get_feature_names_out()

    document_frequency = (matrix > 0).sum(axis=0).A1
    global_mean = matrix.mean(axis=0).A1
    candidates: dict[str, dict] = {}

    for category in sorted(frame["label"].unique()):
        mask = (frame["label"] == category).to_numpy()
        if not mask.any():
            continue

        # Rank by how concentrated a term is in this category rather than by
        # raw TF-IDF. Boilerplate like "management" scores highly everywhere,
        # so its ratio sits near 1 and it drops out; "phlebotomy" does not.
        mean_scores = matrix[mask].mean(axis=0).A1
        ratio = mean_scores / (global_mean + 1e-9)
        ranked = ratio.argsort()[::-1][: top_per_category * 6]

        kept = 0
        for index in ranked:
            if kept >= top_per_category:
                break
            if ratio[index] < _MIN_DISCRIMINATIVENESS:
                break
            term = str(terms[index]).strip()
            if not _SKILL_TOKEN.match(term) or term in _SKILL_STOPWORDS:
                continue
            if any(word in _SKILL_STOPWORDS for word in term.split()):
                continue
            entry = candidates.setdefault(
                term,
                {
                    "term": term,
                    "status": "confirmed" if term in seed_lower else "candidate",
                    "document_frequency": int(document_frequency[index]),
                    "discriminativeness": round(float(ratio[index]), 2),
                    "categories": [],
                },
            )
            if category not in entry["categories"]:
                entry["categories"].append(category)
            kept += 1

    confirmed = [
        {
            "term": skill.lower(),
            "display": skill,
            "status": "confirmed",
            "document_frequency": None,
            "categories": [],
        }
        for skill in seed
    ]
    mined = sorted(
        (entry for entry in candidates.values() if entry["term"] not in seed_lower),
        key=lambda entry: entry["discriminativeness"],
        reverse=True,
    )
    for entry in mined:
        entry["display"] = entry["term"]

    vocabulary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_rows": int(len(frame)),
        "seed_terms": len(confirmed),
        "mined_terms": len(mined),
        "terms": confirmed + mined,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    destination = PROCESSED_DIR / SKILL_VOCAB_FILE
    destination.write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    vocabulary["path"] = str(destination)
    return vocabulary


def load_model(name: str) -> Pipeline:
    """
    Load a previously trained pipeline.

    Args:
        name: Model base name, e.g. ``text_emotion``.

    Returns:
        The fitted pipeline.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    path = Path(MODEL_DIR) / f"{name}.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"Model {name} not trained yet. Run: "
            f"python -m scripts.datasets.cli train"
        )
    return joblib.load(path)
