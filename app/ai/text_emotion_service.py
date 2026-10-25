"""
Text-side emotion analysis over answer transcripts.

`EmotionDetectionService` reads the candidate's face and never sees a word they
said; when there is no webcam it produces nothing at all. This service covers
that gap by classifying the transcript itself, using the TF-IDF model trained
by ``scripts/datasets/train.py`` on the Kaggle emotions corpus.

The model file is optional. Where it is missing - a fresh checkout that never
ran the dataset pipeline - every entry point degrades to returning ``None`` and
the caller keeps its previous behaviour.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from app.utils.constants import (
    EMOTION_HAPPY,
    EMOTION_NEUTRAL,
    EMOTION_NERVOUS,
    EMOTION_SAD,
)

MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "models" / "text_emotion.joblib"
)

# The corpus labels six emotions; EmotionAnalysis stores four. Anger and fear
# both read as nervousness in an interview setting, which is the closest honest
# mapping, and surprise is treated as neutral rather than positive.
CORPUS_EMOTION_TO_APP: dict[str, str] = {
    "joy": EMOTION_HAPPY,
    "love": EMOTION_HAPPY,
    "surprise": EMOTION_NEUTRAL,
    "neutral": EMOTION_NEUTRAL,
    "sadness": EMOTION_SAD,
    "sad": EMOTION_SAD,
    "anger": EMOTION_NERVOUS,
    "fear": EMOTION_NERVOUS,
}

APP_EMOTIONS = (EMOTION_HAPPY, EMOTION_NEUTRAL, EMOTION_SAD, EMOTION_NERVOUS)

# Below this many words the prediction is not worth trusting.
MIN_WORDS = 4


class TextEmotionService:
    """Classify interview answers by emotional tone."""

    _model = None
    _load_attempted = False
    _lock = threading.Lock()

    @classmethod
    def available(cls) -> bool:
        """Report whether the trained model could be loaded."""
        return cls._get_model() is not None

    @classmethod
    def _get_model(cls):
        """
        Load and cache the trained pipeline.

        Returns:
            The fitted pipeline, or ``None`` if it is missing or unreadable.
        """
        if cls._load_attempted:
            return cls._model

        with cls._lock:
            if cls._load_attempted:
                return cls._model
            cls._load_attempted = True
            try:
                import joblib

                if MODEL_PATH.is_file():
                    cls._model = joblib.load(MODEL_PATH)
            except Exception:  # noqa: BLE001 - a missing model must never break an answer
                cls._model = None
        return cls._model

    @staticmethod
    def analyze_text(text: str) -> Optional[dict]:
        """
        Score one transcript across the four stored emotions.

        Args:
            text: The answer transcript.

        Returns:
            A dict shaped for ``EmotionDetectionService.save_analysis`` - the
            four ``*_score`` floats on a 0-100 scale plus ``dominant_emotion``
            and ``confidence_score`` - or ``None`` when the model is
            unavailable or the text is too short to judge.
        """
        if not text or len(text.split()) < MIN_WORDS:
            return None

        model = TextEmotionService._get_model()
        if model is None:
            return None

        try:
            probabilities = model.predict_proba([text])[0]
            classes = model.named_steps["clf"].classes_
        except Exception:  # noqa: BLE001 - never fail an interview on this
            return None

        buckets = {emotion: 0.0 for emotion in APP_EMOTIONS}
        for label, probability in zip(classes, probabilities):
            bucket = CORPUS_EMOTION_TO_APP.get(str(label).lower(), EMOTION_NEUTRAL)
            buckets[bucket] += float(probability)

        dominant = max(buckets, key=buckets.get)
        results = {
            f"{emotion.lower()}_score": round(value * 100, 2)
            for emotion, value in buckets.items()
        }
        results["dominant_emotion"] = dominant
        # Composure, not positivity: a calm neutral answer reads as confident,
        # a nervous one does not.
        results["confidence_score"] = round(
            (buckets[EMOTION_HAPPY] + buckets[EMOTION_NEUTRAL]) * 100, 2
        )
        results["source"] = "text"
        return results

    @staticmethod
    def blend_with_visual(
        visual: Optional[dict], textual: Optional[dict], text_weight: float = 0.4
    ) -> Optional[dict]:
        """
        Combine facial and textual emotion into one result.

        Args:
            visual: Result from ``EmotionDetectionService``, or ``None``.
            textual: Result from :meth:`analyze_text`, or ``None``.
            text_weight: Share given to the text signal when both are present.
                Defaults below 0.5 because the visual pipeline also supplies
                eye contact and attention, which text cannot.

        Returns:
            The merged result, whichever single result exists, or ``None``.
        """
        if visual is None:
            return textual
        if textual is None:
            return visual

        merged = dict(visual)
        visual_weight = 1.0 - text_weight
        for emotion in APP_EMOTIONS:
            key = f"{emotion.lower()}_score"
            merged[key] = round(
                visual.get(key, 0.0) * visual_weight + textual.get(key, 0.0) * text_weight,
                2,
            )

        scores = {emotion: merged[f"{emotion.lower()}_score"] for emotion in APP_EMOTIONS}
        merged["dominant_emotion"] = max(scores, key=scores.get)
        merged["confidence_score"] = round(
            visual.get("confidence_score", 0.0) * visual_weight
            + textual.get("confidence_score", 0.0) * text_weight,
            2,
        )
        merged["source"] = "visual+text"
        return merged
