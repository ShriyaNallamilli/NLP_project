"""
Registry of Kaggle NLP datasets used by InterviewIQ.

Every entry is declarative: nothing here touches the network. The slugs are
resolved and validated at run time by :mod:`scripts.datasets.kaggle_client`,
so a renamed or removed dataset surfaces as an explicit verification failure
rather than a silent empty download.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repository layout ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"

# Canonical task names used by the loaders and trainers.
TASK_RESUME_CLASSIFICATION = "resume-classification"
TASK_TEXT_EMOTION = "text-emotion"
TASK_SENTIMENT = "sentiment"
TASK_INTERVIEW_QUESTIONS = "interview-questions"


@dataclass(frozen=True)
class DatasetSpec:
    """
    A single Kaggle dataset this project consumes.

    Attributes:
        key: Short local name; also the folder name under ``data/raw``.
        slug: Kaggle identifier in ``owner/dataset-name`` form.
        title: Human readable dataset title.
        task: One of the ``TASK_*`` constants.
        why: What this dataset is used for inside InterviewIQ.
        download_files: Exact archive member paths to fetch instead of the whole
            dataset. Several of these datasets bundle hundreds of megabytes of
            source PDFs next to the CSV that actually carries the text, so
            naming the CSV here avoids the bulk. Empty means fetch everything.
        file_hints: Substrings used to pick the payload file after unzip.
            The first hint that matches a downloaded file wins; if none match,
            the loader falls back to the largest CSV in the folder.
        text_columns: Candidate column names holding the free text.
        label_columns: Candidate column names holding the label.
        optional: Skipped by ``download --all`` unless ``--include-optional``.
            Used for corpora that are large or only loosely related.
    """

    key: str
    slug: str
    title: str
    task: str
    why: str
    download_files: tuple[str, ...] = field(default_factory=tuple)
    file_hints: tuple[str, ...] = field(default_factory=tuple)
    text_columns: tuple[str, ...] = field(default_factory=tuple)
    label_columns: tuple[str, ...] = field(default_factory=tuple)
    optional: bool = False

    @property
    def raw_dir(self) -> Path:
        """Folder the archive is unzipped into."""
        return RAW_DIR / self.key


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="resumes_large",
        slug="snehaanbhawal/resume-dataset",
        title="Resume Dataset (~2.4k resumes, 24 job categories)",
        task=TASK_RESUME_CLASSIFICATION,
        why=(
            "Real resume text with job-category labels. Feeds the resume "
            "category classifier and the mined skill vocabulary that widens "
            "ResumeParserService.COMMON_SKILLS."
        ),
        # The dataset also ships ~2.4k source PDFs under data/data/<CATEGORY>/;
        # Resume.csv already contains their extracted text.
        download_files=("Resume/Resume.csv",),
        file_hints=("Resume.csv", "resume"),
        text_columns=("Resume_str", "Resume", "resume_text", "Resume_html"),
        label_columns=("Category", "category"),
    ),
    DatasetSpec(
        key="resumes_classic",
        # gauravduttakiit/resume-dataset carries the same file but now 403s;
        # verified 2026-09-14 against this mirror instead.
        slug="jillanisofttech/updated-resume-dataset",
        title="Updated Resume Dataset (~960 resumes, 25 categories)",
        task=TASK_RESUME_CLASSIFICATION,
        why=(
            "Smaller, cleaner resume corpus. Used together with resumes_large "
            "for a more balanced label distribution, and as a fallback when "
            "the larger set is unavailable."
        ),
        file_hints=("UpdatedResumeDataSet.csv", "resume"),
        text_columns=("Resume", "Resume_str", "resume_text"),
        label_columns=("Category", "category"),
    ),
    DatasetSpec(
        key="emotions_text",
        slug="praveengovi/emotions-dataset-for-nlp",
        title="Emotions dataset for NLP (~20k sentences, 6 emotions)",
        task=TASK_TEXT_EMOTION,
        why=(
            "Sentence-level emotion labels (joy, sadness, anger, fear, love, "
            "surprise). Trains a text-side emotion classifier over answer "
            "transcripts, complementing the OpenCV facial analysis in "
            "app/ai/emotion_service.py, which sees no text at all."
        ),
        file_hints=("train.txt", "val.txt", "test.txt", ".txt"),
        text_columns=("text", "Text", "sentence"),
        label_columns=("label", "emotion", "Emotion"),
    ),
    DatasetSpec(
        key="interview_questions",
        slug="syedmharis/software-engineering-interview-questions-dataset",
        title="Software Engineering Interview Questions",
        task=TASK_INTERVIEW_QUESTIONS,
        why=(
            "Real interview questions with answers. The only corpus here that "
            "could back a seeded question bank, which would need a new table: "
            "questions.interview_id is NOT NULL, so today a question cannot "
            "exist outside a session. See data/README.md."
        ),
        file_hints=("Software Questions.csv", ".csv"),
        text_columns=("Question", "question"),
        label_columns=("Category", "category", "Difficulty", "Answer"),
    ),
    DatasetSpec(
        key="sentiment140",
        slug="kazanova/sentiment140",
        title="Sentiment140 (1.6M labelled tweets)",
        task=TASK_SENTIMENT,
        why=(
            "Large-scale polarity corpus. Optional: useful as a pre-training / "
            "sanity baseline for tone detection, but tweets are stylistically "
            "far from interview answers, so it is not used for seeding."
        ),
        file_hints=("training.1600000", ".csv"),
        text_columns=("text",),
        label_columns=("target", "polarity"),
        optional=True,
    ),
)

DATASETS_BY_KEY: dict[str, DatasetSpec] = {spec.key: spec for spec in DATASETS}


def get(key: str) -> DatasetSpec:
    """
    Look up a dataset spec by key.

    Args:
        key: Local dataset key, e.g. ``resumes_large``.

    Returns:
        The matching :class:`DatasetSpec`.

    Raises:
        KeyError: If the key is not registered.
    """
    try:
        return DATASETS_BY_KEY[key]
    except KeyError:
        known = ", ".join(sorted(DATASETS_BY_KEY))
        raise KeyError(f"Unknown dataset '{key}'. Known keys: {known}") from None


def select(keys: list[str] | None, include_optional: bool = False) -> list[DatasetSpec]:
    """
    Resolve a list of dataset keys, defaulting to the whole registry.

    Args:
        keys: Explicit keys, or ``None`` for every registered dataset.
        include_optional: Keep specs flagged ``optional`` when ``keys`` is None.

    Returns:
        The selected specs, in registry order.
    """
    if keys:
        return [get(key) for key in keys]
    return [spec for spec in DATASETS if include_optional or not spec.optional]


def ensure_dirs() -> None:
    """Create the data directories if they do not exist yet."""
    for directory in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
