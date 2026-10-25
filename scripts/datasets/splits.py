"""
Deterministic train / validation / test splits.

Every module that touches a corpus - training, baselines, evaluation, error
analysis - takes its split from here, so the same row always lands in the same
fold regardless of which command ran. That is what keeps the test set clean:
tuning happens against validation, and test is read once at the end.

The split is a pure function of (texts, labels, seed), not of call order, so
re-running any single command reproduces the identical partition.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

# Held-out fractions. The remainder is training data.
TEST_FRACTION = 0.15
VAL_FRACTION = 0.15
RANDOM_SEED = 42

# A class needs enough rows to appear in all three folds after stratification.
MIN_ROWS_PER_LABEL = 5


@dataclass(frozen=True)
class Split:
    """
    One corpus partitioned into three disjoint frames.

    Attributes:
        train: Rows to fit on.
        val: Rows to tune and compare models against.
        test: Rows read only for the final reported number.
        dropped_labels: Classes removed for having too few rows.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    dropped_labels: tuple[str, ...]

    @property
    def sizes(self) -> dict[str, int]:
        """Row count per fold."""
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }

    def describe(self) -> str:
        """Render the split as a one-line summary."""
        sizes = self.sizes
        total = sum(sizes.values())
        return (
            f"{total} rows -> train {sizes['train']} / "
            f"val {sizes['val']} / test {sizes['test']}; "
            f"{self.train['label'].nunique()} classes"
            + (f"; dropped {len(self.dropped_labels)} rare" if self.dropped_labels else "")
        )


def drop_rare_labels(
    frame: pd.DataFrame, minimum: int = MIN_ROWS_PER_LABEL
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """
    Remove classes too small to stratify across three folds.

    Args:
        frame: Canonical frame with a ``label`` column.
        minimum: Smallest acceptable class size.

    Returns:
        The filtered frame and the labels that were dropped.
    """
    counts = frame["label"].value_counts()
    keep = counts[counts >= minimum].index
    dropped = tuple(sorted(str(label) for label in counts[counts < minimum].index))
    filtered = frame[frame["label"].isin(keep)].reset_index(drop=True)
    return filtered, dropped


def make_split(
    frame: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
    val_fraction: float = VAL_FRACTION,
    seed: int = RANDOM_SEED,
) -> Split:
    """
    Partition a corpus into stratified train / validation / test folds.

    Args:
        frame: Canonical frame with ``text`` and ``label`` columns.
        test_fraction: Share held out for the final test set.
        val_fraction: Share held out for tuning and model comparison.
        seed: Random seed controlling the partition.

    Returns:
        The :class:`Split`.

    Raises:
        ValueError: If fewer than two classes survive filtering.
    """
    usable, dropped = drop_rare_labels(frame.dropna(subset=["text", "label"]))
    if usable["label"].nunique() < 2:
        raise ValueError(
            f"Need at least 2 classes with {MIN_ROWS_PER_LABEL}+ rows; got "
            f"{usable['label'].nunique()} from {len(frame)} rows."
        )

    # First carve off test, then split the remainder into train and validation,
    # so the test fraction is exact regardless of the validation fraction.
    train_val, test = train_test_split(
        usable,
        test_size=test_fraction,
        random_state=seed,
        stratify=usable["label"],
    )
    relative_val = val_fraction / (1.0 - test_fraction)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val["label"],
    )

    return Split(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test.reset_index(drop=True),
        dropped_labels=dropped,
    )


def fit_frame(split: Split, include_val: bool = False) -> pd.DataFrame:
    """
    Return the rows a final model should be fitted on.

    Args:
        split: The partition.
        include_val: Fold validation back in. Correct only once tuning is
            finished - the reported test number stays honest either way, but
            validation scores become meaningless afterwards.

    Returns:
        The training frame, optionally concatenated with validation.
    """
    if not include_val:
        return split.train
    return pd.concat([split.train, split.val], ignore_index=True)
