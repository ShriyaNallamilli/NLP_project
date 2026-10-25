"""
Resume parsing service using pdfplumber and AI.
"""
import json
import os
import re
from pathlib import Path
from typing import Optional

import pdfplumber
from flask import current_app

from app.ai.gemini_service import GeminiService
from app.extensions import db
from app.models import Resume, Skill
from app.utils import nlp
from app.utils.helpers import extract_email, extract_phone, save_upload_file


class ResumeParserService:
    """Parse and analyze uploaded PDF resumes."""

    COMMON_SKILLS: list[str] = [
        "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Go", "Rust",
        "HTML", "CSS", "React", "Angular", "Vue", "Node.js", "Express",
        "Django", "Flask", "Spring", "Spring Boot", "Hibernate",
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "Redis",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
        "Git", "Linux", "REST API", "GraphQL", "Microservices",
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
        "Data Science", "Pandas", "NumPy", "Scikit-learn", "NLP",
        "Computer Vision", "OpenCV", "Cyber Security", "Networking",
        "Agile", "Scrum", "CI/CD", "Jenkins", "JIRA",
        "Communication", "Leadership", "Problem Solving", "Teamwork",
    ]

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """
        Extract text content from PDF file.

        Args:
            file_path: Absolute path to PDF.

        Returns:
            Extracted text string.
        """
        text_parts: list[str] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as exc:
            current_app.logger.error(f"PDF extraction failed: {exc}")
        return "\n".join(text_parts)

    @staticmethod
    def extract_name(text: str) -> Optional[str]:
        """Extract candidate name from resume text."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:5]:
            if "@" not in line and len(line.split()) <= 4 and len(line) < 50:
                if not re.search(r"\d{3}", line):
                    return line
        return None

    @staticmethod
    def extract_section(text: str, section_keywords: list[str]) -> str:
        """
        Extract resume section by keyword headers.

        Args:
            text: Full resume text.
            section_keywords: Section header keywords.

        Returns:
            Section content string.
        """
        pattern = (
            r"(?i)("
            + "|".join(section_keywords)
            + r")\s*[:\-]?\s*\n?(.*?)(?=\n\s*(?:"
            + "|".join(
                [
                    "education", "experience", "skills", "projects",
                    "certifications", "summary", "objective", "work",
                ]
            )
            + r")\s*[:\-]|\Z)"
        )
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(2).strip()[:2000]

        return ResumeParserService._section_after_heading(text, section_keywords)

    @staticmethod
    def _section_after_heading(text: str, section_keywords: list[str]) -> str:
        """
        Recover a section when the header regex finds nothing.

        The regex above assumes the section header sits on its own line.
        Plenty of resumes run the header into the first sentence instead, so
        this takes the sentences following the heading word and stops at the
        next known heading.

        Args:
            text: Full resume text.
            section_keywords: Section header keywords.

        Returns:
            Section content string, or "" when no heading is found.
        """
        all_sentences = nlp.sentences(text)
        if not all_sentences:
            return ""

        wanted = {k.lower() for k in section_keywords}
        other_headings = {
            "education",
            "experience",
            "skills",
            "projects",
            "certifications",
            "summary",
            "objective",
            "work",
        } - wanted

        collected: list[str] = []
        capturing = False
        for sentence in all_sentences:
            opening = " ".join(nlp.word_tokens(sentence)[:4])
            if not capturing:
                if any(keyword in opening for keyword in wanted):
                    capturing = True
                    collected.append(sentence)
                continue
            if any(heading in opening for heading in other_headings):
                break
            collected.append(sentence)

        return " ".join(collected).strip()[:2000]

    # Vocabulary mined from the Kaggle resume corpus by
    # scripts/datasets/cli.py mine-skills. Absent until that has been run.
    SKILL_VOCAB_PATH = (
        Path(__file__).resolve().parents[2] / "data" / "processed" / "skills_vocab.json"
    )

    # Mined terms start as "candidate" and are noisy; promote a reviewed term to
    # "approved" in skills_vocab.json to have it matched. Set
    # SKILL_VOCAB_INCLUDE_CANDIDATES=1 to match unreviewed terms too.
    SKILL_VOCAB_STATUSES = frozenset({"confirmed", "approved"})

    _skill_vocab_cache: Optional[list[str]] = None

    @classmethod
    def skill_vocabulary(cls) -> list[str]:
        """
        Return the skill terms used for matching, loading the mined file once.

        Falls back to the hard-coded COMMON_SKILLS whenever the mined
        vocabulary is missing or unreadable, so resume parsing keeps working on
        a checkout where the datasets were never downloaded.

        Returns:
            Skill strings to search for in resume text.
        """
        if cls._skill_vocab_cache is not None:
            return cls._skill_vocab_cache

        vocabulary = list(cls.COMMON_SKILLS)
        statuses = set(cls.SKILL_VOCAB_STATUSES)
        if os.getenv("SKILL_VOCAB_INCLUDE_CANDIDATES", "").strip() in {"1", "true", "True"}:
            statuses.add("candidate")

        try:
            if cls.SKILL_VOCAB_PATH.is_file():
                payload = json.loads(cls.SKILL_VOCAB_PATH.read_text(encoding="utf-8"))
                mined = [
                    entry.get("display") or entry["term"]
                    for entry in payload.get("terms", [])
                    if entry.get("status") in statuses
                ]
                if mined:
                    vocabulary = mined
        except (OSError, ValueError, KeyError, TypeError):
            # A malformed vocabulary must never break an upload.
            vocabulary = list(cls.COMMON_SKILLS)

        # Longest first so "Spring Boot" is preferred over "Spring".
        cls._skill_vocab_cache = sorted(
            dict.fromkeys(vocabulary), key=len, reverse=True
        )
        return cls._skill_vocab_cache

    _skill_pattern_cache: Optional[list[tuple[str, "re.Pattern[str]"]]] = None

    @classmethod
    def _skill_patterns(cls) -> list[tuple[str, "re.Pattern[str]"]]:
        """
        Compile one boundary-anchored pattern per vocabulary term.

        Plain substring matching makes short terms catastrophic: "Go" matches
        inside "going" and "category", and "R" matches everything. The
        lookarounds below test for an alphanumeric neighbour rather than using
        \\b, because \\b does not behave at the edges of terms like "C++",
        ".NET" or "Node.js".

        Returns:
            (display term, compiled pattern) pairs.
        """
        if cls._skill_pattern_cache is not None:
            return cls._skill_pattern_cache

        patterns: list[tuple[str, "re.Pattern[str]"]] = []
        for skill in cls.skill_vocabulary():
            escaped = re.escape(skill.lower())
            patterns.append(
                (
                    skill,
                    re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.IGNORECASE),
                )
            )
        cls._skill_pattern_cache = patterns
        return patterns

    @staticmethod
    def extract_skills(text: str) -> list[str]:
        """
        Extract skills from resume text.

        Args:
            text: Resume text.

        Returns:
            List of matched skills.
        """
        found: list[str] = []
        text_lower = text.lower()
        for skill, pattern in ResumeParserService._skill_patterns():
            if pattern.search(text_lower):
                found.append(skill)
        return list(dict.fromkeys(found))

    # Tags that can open or continue a skill-like noun phrase. Adjectives are
    # allowed at the front ("distributed systems") but never on their own.
    _PHRASE_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS", "JJ"})
    _NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})

    @staticmethod
    def extract_keyword_phrases(text: str, limit: int = 40) -> list[str]:
        """
        Pull candidate noun phrases out of resume text.

        This is what surfaces skills the fixed vocabulary has never heard of.
        It is a suggestion channel, not a match: phrases are ranked by how
        often they occur, and the caller keeps them separate from confirmed
        vocabulary hits.

        Returns an empty list when the POS tagger is unavailable, because
        there is no honest regex substitute for tagging.

        Args:
            text: Resume text.
            limit: Maximum number of phrases to return.

        Returns:
            Candidate phrases, most frequent first.
        """
        stops = nlp.stopword_set()
        counts: dict[str, int] = {}
        tagged_anything = False

        # Chunking runs per sentence for two reasons: the perceptron tagger
        # expects a sentence rather than a bag of words, and word_tokens()
        # drops the full stops, so a document-wide pass would run the last
        # noun of one sentence into the first noun of the next.
        for sentence in nlp.sentences(text):
            tagged = nlp.pos_tags(nlp.word_tokens(sentence))
            if not tagged:
                continue
            tagged_anything = True

            current: list[str] = []
            current_has_noun = False

            def flush() -> None:
                """Keep the phrase built so far when it looks like a skill."""
                nonlocal current, current_has_noun
                # One word is mostly noise and five is a sentence, so only
                # two-to-four-word phrases carrying a noun are kept.
                if 1 < len(current) <= 4 and current_has_noun:
                    phrase = " ".join(current)
                    if len(phrase) >= 4:
                        counts[phrase] = counts.get(phrase, 0) + 1
                current = []
                current_has_noun = False

            for token, tag in tagged:
                if tag in ResumeParserService._PHRASE_TAGS and token not in stops:
                    current.append(token)
                    current_has_noun = current_has_noun or (
                        tag in ResumeParserService._NOUN_TAGS
                    )
                    continue
                flush()
            flush()

        if not tagged_anything:
            return []

        return sorted(counts, key=lambda p: (-counts[p], p))[:limit]

    @staticmethod
    def extract_candidate_skills(
        text: str, known_skills: list[str], limit: int = 15
    ) -> list[str]:
        """
        Suggest skill phrases the fixed vocabulary did not already match.

        Args:
            text: Resume text.
            known_skills: Skills already matched by the vocabulary.
            limit: Maximum number of suggestions.

        Returns:
            Candidate skill phrases, most frequent first.
        """
        seen = {skill.lower() for skill in known_skills}
        suggestions: list[str] = []
        for phrase in ResumeParserService.extract_keyword_phrases(text):
            lowered = phrase.lower()
            # Skip anything a confirmed skill already covers.
            if lowered in seen or any(known in lowered for known in seen):
                continue
            suggestions.append(phrase)
            seen.add(lowered)
            if len(suggestions) >= limit:
                break
        return suggestions

    @staticmethod
    def process_upload(file, user_id: int, domain: str = "Python") -> Resume:
        """
        Process uploaded resume file end-to-end.

        Args:
            file: Uploaded file object.
            user_id: Owner user ID.
            domain: Target domain for skill gap analysis.

        Returns:
            Created Resume model instance.
        """
        relative_path = save_upload_file(file, subfolder="resumes")
        full_path = os.path.join(
            current_app.config["UPLOAD_FOLDER"], relative_path
        )

        raw_text = ResumeParserService.extract_text_from_pdf(full_path)
        name = ResumeParserService.extract_name(raw_text)
        email = extract_email(raw_text)
        phone = extract_phone(raw_text)
        education = ResumeParserService.extract_section(
            raw_text, ["education", "academic", "qualification"]
        )
        experience = ResumeParserService.extract_section(
            raw_text, ["experience", "work experience", "employment", "work history"]
        )
        projects = ResumeParserService.extract_section(
            raw_text, ["projects", "personal projects", "academic projects"]
        )
        certifications = ResumeParserService.extract_section(
            raw_text, ["certifications", "certificates", "licenses"]
        )
        skills = ResumeParserService.extract_skills(raw_text)
        candidate_skills = ResumeParserService.extract_candidate_skills(
            raw_text, skills
        )

        summary = GeminiService.generate_resume_summary(raw_text, skills)
        missing = GeminiService.suggest_missing_skills(skills, domain)
        missing_skills_text = ", ".join(missing)

        Resume.query.filter_by(user_id=user_id, is_primary=True).update(
            {"is_primary": False}
        )

        resume = Resume(
            user_id=user_id,
            file_name=file.filename,
            file_path=relative_path,
            raw_text=raw_text,
            name=name,
            email=email,
            phone=phone,
            education=education,
            experience=experience,
            projects=projects,
            certifications=certifications,
            summary=summary,
            missing_skills=missing_skills_text,
            is_primary=True,
        )
        db.session.add(resume)
        db.session.flush()

        for skill_name in skills:
            db.session.add(
                Skill(resume_id=resume.id, name=skill_name, category="Technical")
            )

        # Kept under a separate category so the UI can present these as
        # guesses instead of mixing them into confirmed vocabulary matches.
        for phrase in candidate_skills:
            db.session.add(
                Skill(resume_id=resume.id, name=phrase, category="Detected")
            )

        db.session.commit()
        return resume

    @staticmethod
    def get_primary_resume(user_id: int) -> Optional[Resume]:
        """Get user's primary resume."""
        return Resume.query.filter_by(user_id=user_id, is_primary=True).first()

    @staticmethod
    def get_user_resumes(user_id: int) -> list:
        """Get all resumes for user."""
        return (
            Resume.query.filter_by(user_id=user_id)
            .order_by(Resume.created_at.desc())
            .all()
        )
