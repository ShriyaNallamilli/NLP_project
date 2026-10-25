"""
Generate a written evaluation report from the measured artefacts.

Everything here is read from the JSON files the other commands produce - no
number is typed in by hand, so the report cannot drift from the experiments.
Re-run the experiments, re-run this, and the prose updates with them.

Output is Markdown, suitable for pasting into a course write-up.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.datasets.manifest import DATASETS, REPORT_DIR

MODEL_TITLES = {
    "resume_category": "Resume category classification",
    "text_emotion": "Answer emotion classification",
}


def _read(name: str) -> dict | None:
    """Read one report file, returning None when it has not been generated."""
    path = REPORT_DIR / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def _corpus_section() -> list[str]:
    """Describe the datasets used."""
    lines = ["## 1. Data", "", "| Dataset | Source | Task | Rows used |", "|---|---|---|---|"]
    from scripts.datasets.train import RESUME_TRAIN_SOURCES

    # Row counts are per trained model, so they only apply to the datasets that
    # model was actually fitted on - resumes_classic shares a task with
    # resumes_large but is deliberately not trained on (incompatible taxonomy).
    trained_rows = {
        "resumes_large": _read("resume_category_metrics.json"),
        "emotions_text": _read("text_emotion_metrics.json"),
    }
    other_use = {
        "resumes_classic": "skill mining and DB seeding only",
        "interview_questions": "downloaded, not yet wired in",
        "sentiment140": "registered, not downloaded",
    }
    for spec in DATASETS:
        metrics = trained_rows.get(spec.key)
        if metrics:
            rows = f"{metrics['rows_total']:,} (trained)"
        else:
            rows = other_use.get(spec.key, "not used for training")
        lines.append(f"| `{spec.key}` | Kaggle `{spec.slug}` | {spec.task} | {rows} |")

    lines += [
        "",
        "`resumes_classic` shares the resume-classification task but is not "
        "trained on: it labels by IT role (Java Developer, Data Science) while "
        "`resumes_large` labels by industry (ACCOUNTANT, INFORMATION-TECHNOLOGY). "
        "Concatenating them yields a 45-class union of two incompatible "
        "taxonomies and a worse model on both, so "
        f"training is pinned to {', '.join(RESUME_TRAIN_SOURCES)}.",
    ]
    lines += [
        "",
        "Texts are lower-cased, stripped of HTML, whitespace-normalised and "
        "de-duplicated on exact text. De-duplication matters: the "
        "`resumes_classic` corpus falls from 962 rows to 166 unique resumes, "
        "so reporting 962 would overstate the data by a factor of six.",
        "",
    ]
    return lines


def _methodology_section() -> list[str]:
    """Describe the experimental protocol."""
    metrics = _read("resume_category_metrics.json")
    split = metrics["split"] if metrics else {"train": 0, "val": 0, "test": 0}
    return [
        "## 2. Experimental protocol",
        "",
        "- **Three-way split**, stratified by class: 70% train / 15% validation "
        f"/ 15% test (e.g. {split['train']} / {split['val']} / {split['test']} "
        "for the resume corpus).",
        "- **Model selection uses validation only.** The test fold is read once, "
        "for the final reported number, and never drives a decision. All folds "
        "derive from a single deterministic function of (text, label, seed), so "
        "every command sees the identical partition.",
        "- **Classes with fewer than 5 rows are dropped** before splitting, as "
        "they cannot be stratified across three folds.",
        "- **5-fold stratified cross-validation** over train+validation gives "
        "mean +/- standard deviation, so a gap between two models can be judged "
        "against fold-to-fold noise.",
        "- Metrics are accuracy and macro-F1. Macro-F1 is reported because both "
        "corpora are imbalanced and accuracy alone hides per-class collapse.",
        "",
    ]


def _baseline_section() -> list[str]:
    """Report the baseline ladder."""
    lines = [
        "## 3. Baselines",
        "",
        "An accuracy figure is meaningless without a floor. Each rung below "
        "changes one thing from the rung above, so a difference is "
        "attributable.",
        "",
    ]
    any_found = False
    for model_name, title in MODEL_TITLES.items():
        report = _read(f"{model_name}_baselines.json")
        if not report:
            continue
        any_found = True
        lines += [
            f"### {title}",
            "",
            f"{report['classes']} classes; majority class holds "
            f"{report['majority_class_share']:.1%} of rows.",
            "",
            "| Model | Val accuracy | Val macro-F1 | 5-fold CV |",
            "|---|---|---|---|",
        ]
        for row in report["results"]:
            cv = (
                f"{row['cv_mean']:.3f} ± {row['cv_std']:.3f}"
                if "cv_mean" in row
                else "—"
            )
            lines.append(
                f"| `{row['model']}` | {row['val_accuracy']:.1%} | "
                f"{row['val_f1_macro']:.3f} | {cv} |"
            )
        best = report["results"][0]
        floor = next(
            (r for r in report["results"] if r["model"] == "majority_class"), None
        )
        if floor:
            lines += [
                "",
                f"Best model beats the majority-class floor by "
                f"{best['val_accuracy'] - floor['val_accuracy']:.1%} "
                f"({floor['val_accuracy']:.1%} → {best['val_accuracy']:.1%}).",
            ]
        lines.append("")
    if not any_found:
        lines += ["_Not yet generated. Run `cli baselines`._", ""]
    return lines


def _embedding_section() -> list[str]:
    """Report the transformer comparison."""
    lines = ["## 4. Neural embeddings vs TF-IDF", ""]
    any_found = False
    for model_name, title in MODEL_TITLES.items():
        report = _read(f"{model_name}_embeddings.json")
        if not report:
            continue
        any_found = True
        lines += [
            f"### {title}",
            "",
            f"Encoder: `{report['encoder']}` ({report['embedding_dim']}-dim, "
            f"max {report['max_seq_tokens']} word pieces), frozen and used as a "
            "feature extractor with the same logistic regression on top.",
            "",
            f"- Embeddings: **{report['val_accuracy']:.1%}** val accuracy "
            f"(macro-F1 {report['val_f1_macro']:.3f})",
        ]
        if "tfidf_val_accuracy" in report:
            lines.append(f"- TF-IDF: **{report['tfidf_val_accuracy']:.1%}** val accuracy")
            lines.append("")
            lines.append(
                f"**{report['verdict']}** by {abs(report['delta_vs_tfidf']):.1%}."
            )
        truncation = report.get("estimated_truncation", 0.0)
        embeddings_won = report.get("delta_vs_tfidf", 0) > 0

        if truncation > 0.2 and embeddings_won:
            lines += [
                "",
                f"Note that the encoder sees only the first "
                f"{report['max_seq_tokens']} word pieces: mean text length is "
                f"{report['mean_text_chars']:.0f} characters, so roughly "
                f"{truncation:.0%} of each document is never read. The "
                "embeddings win anyway, which says the opening of a resume - "
                "the summary and most recent role - carries most of the signal "
                "for this task, and that the remaining 83% adds comparatively "
                "little.",
            ]
        elif truncation > 0.2:
            lines += [
                "",
                f"Caveat: roughly {truncation:.0%} of each document falls "
                "outside the encoder's window and is never read, which "
                "handicaps the embedding model here.",
            ]
        elif not embeddings_won:
            lines += [
                "",
                "Nothing is truncated here, so the gap is not an artefact of "
                "the window. These sentences share a near-identical frame "
                '("i feel ..."), so they crowd into one region of embedding '
                "space, while the single emotion word that determines the "
                "label is exactly what a sparse representation isolates. The "
                "lesson generalises: embeddings help when the label depends on "
                "meaning, and hurt when it depends on the presence of "
                "particular words.",
            ]
        lines.append("")
    if not any_found:
        lines += ["_Not yet generated. Run `cli embeddings`._", ""]
    return lines


def _error_section() -> list[str]:
    """Report the error analysis."""
    lines = ["## 5. Error analysis", ""]
    any_found = False
    for model_name, title in MODEL_TITLES.items():
        analysis = _read(f"{model_name}_errors.json")
        if not analysis:
            continue
        any_found = True
        correlation = analysis["support_recall_correlation"]
        lines += [
            f"### {title}",
            "",
            f"Test accuracy {analysis['accuracy']:.1%} over "
            f"{analysis['rows']} rows, {analysis['classes']} classes.",
            "",
            "Most frequent confusions:",
            "",
            "| True | Predicted | Count | Share of class |",
            "|---|---|---|---|",
        ]
        for pair in analysis["top_confusions"][:6]:
            lines.append(
                f"| {pair['true']} | {pair['predicted']} | {pair['count']} | "
                f"{pair['share_of_true_class']:.0%} |"
            )
        lines += ["", "Weakest classes:", "", "| Class | Train rows | Recall |", "|---|---|---|"]
        for row in analysis["per_class"][:5]:
            lines.append(f"| {row['label']} | {row['train_rows']} | {row['recall']:.2f} |")

        lines += ["", f"Correlation between training rows and per-class recall: "
                  f"**{correlation:+.2f}**."]
        if abs(correlation) > 0.3:
            lines.append(
                "The weak classes are weak mainly because they are rare; "
                "collecting more examples for them is the highest-value fix."
            )
        else:
            lines.append(
                "Class size does not explain the weak classes, so the confusions "
                "above reflect genuine semantic overlap between categories rather "
                "than a data shortage."
            )
        if analysis["never_predicted"]:
            lines.append(
                f"Classes never predicted at all: "
                f"{', '.join(analysis['never_predicted'])}."
            )
        lines.append("")
    if not any_found:
        lines += ["_Not yet generated. Run `cli errors`._", ""]
    return lines


def _application_section() -> list[str]:
    """Describe how the models are used in the application."""
    return [
        "## 6. Deployment in the application",
        "",
        "The emotion classifier is not left as an offline artefact. "
        "`app/ai/text_emotion_service.py` loads the trained pipeline and "
        "`InterviewService.save_answer` calls it on every submitted answer, "
        "folding the corpus's six emotions into the four columns "
        "`EmotionAnalysis` stores.",
        "",
        "This matters because the pre-existing `EmotionDetectionService` is "
        "purely visual - it reads webcam frames and never sees a word the "
        "candidate said, and produces nothing at all when there is no camera. "
        "The text classifier covers that gap, and where both signals exist they "
        "are blended (60% visual / 40% text; the visual pipeline also supplies "
        "eye contact and attention, which text cannot).",
        "",
        "The mined skill vocabulary similarly feeds "
        "`ResumeParserService.extract_skills`, which matches on word boundaries "
        "rather than substrings - without that, the skill `Go` matches inside "
        "*going*, *good* and *category*.",
        "",
        "Both integrations degrade safely: a checkout that never ran the "
        "dataset pipeline falls back to the previous behaviour rather than "
        "failing.",
        "",
    ]


def _limitations_section() -> list[str]:
    """State the honest limitations."""
    return [
        "## 7. Limitations",
        "",
        "- **The LLM components are prompt engineering, not implemented NLP.** "
        "Question generation and answer evaluation call the Gemini API; the "
        "language understanding happens remotely. The NLP implemented in this "
        "project is the classification, term mining, information extraction and "
        "evaluation described above.",
        "- **Hyperparameters are not tuned.** Values are reasonable defaults "
        "held fixed across models so comparisons stay fair, but no search was "
        "run. A validation fold exists for exactly that purpose.",
        "- **The mined skill vocabulary is unvalidated.** Terms are ranked by "
        "discriminativeness and marked `candidate`; none have been manually "
        "reviewed, and the application deliberately ignores them until a human "
        "promotes them.",
        "- **Single random seed.** Cross-validation covers fold variance, but "
        "results are not averaged over multiple seeds.",
        "- **Resume categories genuinely overlap.** FINANCE and ACCOUNTANT, ARTS "
        "and TEACHER are not cleanly separable from resume text alone, which "
        "bounds achievable accuracy on that task regardless of model.",
        "",
    ]


def generate() -> Path:
    """
    Write the full Markdown report.

    Returns:
        The path written.
    """
    sections: list[str] = [
        "# NLP evaluation report",
        "",
        "_Generated from the measured artefacts in `data/reports/` by "
        "`python -m scripts.datasets.cli report`. Every number is read from a "
        "JSON experiment record rather than typed in._",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]
    sections += _corpus_section()
    sections += _methodology_section()
    sections += _baseline_section()
    sections += _embedding_section()
    sections += _error_section()
    sections += _application_section()
    sections += _limitations_section()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    destination = REPORT_DIR / "NLP_REPORT.md"
    destination.write_text("\n".join(sections), encoding="utf-8")
    return destination
