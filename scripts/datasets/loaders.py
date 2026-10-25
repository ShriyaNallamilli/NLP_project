"""
Normalise raw Kaggle downloads into canonical DataFrames.

Every loader returns the same three columns so downstream trainers and seeders
never care which corpus a row came from:

    text    str   the document / sentence
    label   str   the target class
    source  str   the dataset key the row came from

Kaggle contributors rename columns and re-zip files freely, so file and column
resolution is hint-driven with fallbacks rather than hard-coded.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

from scripts.datasets.manifest import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    TASK_RESUME_CLASSIFICATION,
    TASK_SENTIMENT,
    TASK_TEXT_EMOTION,
    DatasetSpec,
    select,
)

# Sentiment140 ships headerless in this fixed column order.
SENTIMENT140_COLUMNS = ["target", "id", "date", "flag", "user", "text"]
SENTIMENT140_LABELS = {0: "negative", 2: "neutral", 4: "positive"}

MIN_TEXT_LENGTH = 20

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


class DatasetNotDownloaded(RuntimeError):
    """Raised when a loader is called before the raw files exist."""


def clean_text(value: object) -> str:
    """
    Strip HTML and collapse whitespace.

    Args:
        value: Any cell value; non-strings become an empty string.

    Returns:
        Normalised single-line text.
    """
    if not isinstance(value, str):
        return ""
    return _WHITESPACE.sub(" ", _HTML_TAG.sub(" ", value)).strip()


def seed_skill_list() -> list[str]:
    """
    Read ``ResumeParserService.COMMON_SKILLS`` straight from the source file.

    Parsing the AST instead of importing keeps this module free of Flask and
    pdfplumber, so the dataset tooling runs in a bare pandas environment.

    Returns:
        The skill strings currently hard-coded in the app, or an empty list if
        the attribute can no longer be found.
    """
    source_file = PROJECT_ROOT / "app" / "resume" / "services.py"
    if not source_file.is_file():
        return []

    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign):
            target = node.target
        elif isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "COMMON_SKILLS" and node.value:
            try:
                return [str(item) for item in ast.literal_eval(node.value)]
            except (ValueError, SyntaxError):
                return []
    return []


def resolve_payload(spec: DatasetSpec) -> list[Path]:
    """
    Pick the data files inside a downloaded dataset folder.

    Args:
        spec: The dataset whose ``raw_dir`` should be inspected.

    Returns:
        Matching files, hint order first, largest tabular file as fallback.

    Raises:
        DatasetNotDownloaded: If the folder is missing or empty.
    """
    if not spec.raw_dir.is_dir():
        raise DatasetNotDownloaded(
            f"Dataset {spec.key} is not downloaded yet. Run: "
            f"python -m scripts.datasets.cli download --datasets {spec.key}"
        )

    files = [
        path
        for path in spec.raw_dir.glob("**/*")
        if path.is_file() and path.name != ".gitkeep"
    ]
    if not files:
        raise DatasetNotDownloaded(f"Dataset folder is empty: {spec.raw_dir}")

    matched: list[Path] = []
    for hint in spec.file_hints:
        for path in sorted(files):
            if hint.lower() in path.name.lower() and path not in matched:
                matched.append(path)
    if matched:
        return matched

    tabular = [p for p in files if p.suffix.lower() in {".csv", ".tsv", ".txt", ".json"}]
    pool = tabular or files
    return [max(pool, key=lambda path: path.stat().st_size)]


def _is_jsonl(path: Path) -> bool:
    """Cheap check for newline-delimited JSON."""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        first = handle.readline().strip()
    return first.startswith("{") and first.endswith("}")


def _read_any(path: Path) -> pd.DataFrame:
    """
    Read a csv / tsv / json / semicolon-txt file into a DataFrame.

    Args:
        path: File to read.

    Returns:
        The parsed frame; malformed rows are skipped rather than fatal.
    """
    suffix = path.suffix.lower()
    if suffix == ".json":
        return pd.read_json(path, lines=_is_jsonl(path))
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding="utf-8", on_bad_lines="skip")
    if suffix == ".txt":
        # The emotions corpus ships as "sentence;label" with no header row.
        return pd.read_csv(
            path,
            sep=";",
            header=None,
            names=["text", "label"],
            encoding="utf-8",
            on_bad_lines="skip",
        )
    try:
        return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column present in the frame, case-insensitively."""
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_dataset(spec: DatasetSpec, max_rows: int | None = None) -> pd.DataFrame:
    """
    Load one dataset into the canonical text / label / source frame.

    Args:
        spec: Dataset to load.
        max_rows: Optional cap applied after cleaning, sampled deterministically.

    Returns:
        The normalised frame, with short and duplicate texts dropped.

    Raises:
        DatasetNotDownloaded: If the raw files are absent.
        ValueError: If no usable text column can be identified.
    """
    frames: list[pd.DataFrame] = []
    for path in resolve_payload(spec):
        raw = _read_any(path)
        if raw.empty:
            continue

        if spec.key == "sentiment140" and len(raw.columns) == len(SENTIMENT140_COLUMNS):
            raw.columns = SENTIMENT140_COLUMNS

        text_column = _pick_column(raw, spec.text_columns)
        label_column = _pick_column(raw, spec.label_columns)
        if text_column is None:
            # Last resort: the widest object column is almost always the text.
            object_columns = [c for c in raw.columns if raw[c].dtype == object]
            if not object_columns:
                continue
            text_column = max(
                object_columns,
                key=lambda column: raw[column].astype(str).str.len().mean(),
            )

        labels = (
            raw[label_column].astype(str).str.strip() if label_column else "unlabelled"
        )
        block = pd.DataFrame(
            {
                "text": raw[text_column].map(clean_text),
                "label": labels,
                "source": spec.key,
            }
        )
        if spec.key == "sentiment140":
            block["label"] = (
                pd.to_numeric(block["label"], errors="coerce")
                .map(SENTIMENT140_LABELS)
                .fillna("unknown")
            )
        frames.append(block)

    if not frames:
        raise ValueError(f"No usable text column found in dataset {spec.key}.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["text"].str.len() > MIN_TEXT_LENGTH]
    combined = combined.drop_duplicates(subset="text").reset_index(drop=True)
    if max_rows is not None and len(combined) > max_rows:
        combined = combined.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return combined


def load_task(
    task: str,
    max_rows: int | None = None,
    sources: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Load and concatenate the downloaded datasets registered for a task.

    Args:
        task: One of the TASK_* constants in :mod:`scripts.datasets.manifest`.
        max_rows: Optional per-dataset row cap.
        sources: Restrict to these dataset keys. Needed whenever two corpora
            share a task but not a label taxonomy - concatenating those would
            silently produce a union of two class vocabularies.

    Returns:
        The combined canonical frame.

    Raises:
        DatasetNotDownloaded: If no dataset for the task has been downloaded.
    """
    specs = [spec for spec in select(None, include_optional=True) if spec.task == task]
    if sources is not None:
        specs = [spec for spec in specs if spec.key in sources]
    frames: list[pd.DataFrame] = []
    problems: list[str] = []

    for spec in specs:
        try:
            frames.append(load_dataset(spec, max_rows=max_rows))
        except (DatasetNotDownloaded, ValueError) as exc:
            problems.append(f"  {spec.key}: {exc}")

    if not frames:
        detail = "\n".join(problems) or "  no datasets registered for this task"
        raise DatasetNotDownloaded(f"No data available for task {task}:\n{detail}")

    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset="text").reset_index(drop=True)


def load_resumes(
    max_rows: int | None = None, sources: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """
    Load the resume corpora (resume text plus job category).

    Args:
        max_rows: Optional per-dataset row cap.
        sources: Dataset keys to include. Pass a single key when training a
            classifier: ``resumes_large`` labels by industry
            (INFORMATION-TECHNOLOGY, ACCOUNTANT) while ``resumes_classic``
            labels by IT role (Java Developer, Data Science), and the two
            taxonomies do not merge. ``None`` loads both, which is what the
            skill mining wants - more text, labels barely used.

    Returns:
        The canonical frame.
    """
    return load_task(TASK_RESUME_CLASSIFICATION, max_rows=max_rows, sources=sources)


def load_text_emotions(max_rows: int | None = None) -> pd.DataFrame:
    """Load the sentence-level emotion corpus."""
    return load_task(TASK_TEXT_EMOTION, max_rows=max_rows)


def load_sentiment(max_rows: int | None = 200_000) -> pd.DataFrame:
    """Load the sentiment corpus, capped by default because it is very large."""
    return load_task(TASK_SENTIMENT, max_rows=max_rows)


def write_processed(frame: pd.DataFrame, name: str) -> Path:
    """
    Persist a canonical frame to ``data/processed/<name>.csv``.

    Args:
        frame: The frame to write.
        name: Base file name without extension.

    Returns:
        The path written.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    destination = PROCESSED_DIR / f"{name}.csv"
    frame.to_csv(destination, index=False, encoding="utf-8")
    return destination
