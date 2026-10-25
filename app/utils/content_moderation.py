"""
Content moderation for interview responses.

Matching runs over normalised NLTK tokens rather than raw substrings, so an
inflected form - "cheating" against a "cheat" entry - is caught instead of
slipping through an exact-match test. Where NLTK is unavailable the check
falls back to the previous regex tokenizing and exact matching, which is
weaker but never blocks an interview.
"""
from app.utils import nlp
from app.utils.constants import (
    MODERATION_CANCELLATION_MESSAGE,
    PROFANITY_WORDS,
    SENSITIVE_WORDS,
)


class ContentModerationService:
    """Detect profanity and sensitive language in candidate responses."""

    @staticmethod
    def check_text(text: str) -> dict:
        """
        Check text for profanity or sensitive words.

        Args:
            text: Candidate response text to validate.

        Returns:
            Dictionary with is_violation flag, matched words, and reason.
        """
        if not text or not text.strip():
            return {
                "is_violation": False,
                "matched_words": [],
                "reason": "",
            }

        tokens = nlp.word_tokens(text)
        # Base forms let "cheated" and "cheating" match a "cheat" entry.
        token_forms = nlp.normalize_tokens(tokens)
        joined = " ".join(tokens)

        matched: list[str] = []

        for word in PROFANITY_WORDS + SENSITIVE_WORDS:
            word_lower = word.lower()
            if " " in word_lower:
                # Phrases are matched against the rejoined token stream, so
                # punctuation between the words does not hide them.
                phrase = " ".join(nlp.word_tokens(word_lower))
                if phrase and phrase in joined:
                    matched.append(word)
                continue
            if word_lower in token_forms:
                matched.append(word)
                continue
            if nlp.normalize_tokens([word_lower]) & token_forms:
                matched.append(word)

        if matched:
            return {
                "is_violation": True,
                "matched_words": matched,
                "reason": MODERATION_CANCELLATION_MESSAGE,
            }

        return {
            "is_violation": False,
            "matched_words": [],
            "reason": "",
        }
