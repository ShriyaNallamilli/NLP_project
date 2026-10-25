"""
Seed smarthire.db from the Kaggle corpora.

Two things are seeded:

    skill vocabulary   data/processed/skills_vocab.json, mined from the resume
                       corpus and loaded at run time by ResumeParserService.
    sample resumes     Resume + Skill rows drawn from the corpus, attached to a
                       dedicated seed account so the app has realistic content.

Seeded resumes are marked with a ``dataset://`` file_path prefix, which is how
``--reset`` finds and removes them without touching real uploads. They are
never marked primary, so a real candidate upload still wins.

The questions table is deliberately not seeded: it requires an interview_id, so
questions only exist inside a session, and none of the registered corpora carry
interview questions anyway. See data/README.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.datasets.manifest import PROCESSED_DIR
from scripts.datasets.train import SKILL_VOCAB_FILE

SEED_PATH_PREFIX = "dataset://"
SEED_USER_EMAIL = "dataset.seed@interviewiq.local"
SEED_USER_NAME = "dataset_seed"

# Used to fill Skill.category, which is a free-text column in the model.
SOFT_SKILL_HINTS = {
    "communication", "leadership", "teamwork", "problem solving", "management",
    "presentation", "collaboration", "mentoring", "negotiation", "planning",
}


def _skill_category(term: str) -> str:
    """Classify a mined term as a soft or technical skill."""
    lowered = term.lower()
    if any(hint in lowered for hint in SOFT_SKILL_HINTS):
        return "Soft"
    return "Technical"


def load_skill_vocabulary() -> dict:
    """
    Read the mined skill vocabulary from disk.

    Returns:
        The vocabulary dict.

    Raises:
        FileNotFoundError: If the mining step has not been run.
    """
    path = PROCESSED_DIR / SKILL_VOCAB_FILE
    if not path.is_file():
        raise FileNotFoundError(
            "Skill vocabulary not built yet. Run: "
            "python -m scripts.datasets.cli mine-skills"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def vocabulary_terms(statuses: frozenset[str] | None = None) -> list[str]:
    """
    Flatten the vocabulary into a plain list of skill strings.

    Args:
        statuses: Entry statuses to keep. Defaults to the same set the app
            matches on, so seeded skills look like what a real upload would
            produce - seeding with unreviewed 'candidate' terms would fill the
            skills table with resume boilerplate the app itself ignores.

    Returns:
        Display forms of the matching vocabulary terms.
    """
    from app.resume.services import ResumeParserService

    wanted = statuses or ResumeParserService.SKILL_VOCAB_STATUSES
    vocabulary = load_skill_vocabulary()
    return [
        entry.get("display", entry["term"])
        for entry in vocabulary["terms"]
        if entry.get("status") in wanted
    ]


def _get_seed_user(db, models):
    """
    Fetch or create the dedicated seed candidate account.

    The account has no password hash and no google_id, so it cannot be signed
    into; it exists only to own the sample resumes.

    Args:
        db: The SQLAlchemy extension instance.
        models: The ``app.models`` module.

    Returns:
        The seed User row.
    """
    user = models.User.query.filter_by(email=SEED_USER_EMAIL).first()
    if user:
        return user

    role = models.Role.query.filter_by(name="candidate").first()
    if role is None:
        role = models.Role(name="candidate", description="Candidate")
        db.session.add(role)
        db.session.flush()

    user = models.User(
        email=SEED_USER_EMAIL,
        username=SEED_USER_NAME,
        full_name="Dataset Seed Account",
        role_id=role.id,
        is_active=False,
        is_verified=False,
    )
    db.session.add(user)
    db.session.flush()
    return user


def reset_seeded(db, models) -> int:
    """
    Delete every previously seeded resume and its skills.

    Args:
        db: The SQLAlchemy extension instance.
        models: The ``app.models`` module.

    Returns:
        The number of resumes removed.
    """
    seeded = models.Resume.query.filter(
        models.Resume.file_path.like(f"{SEED_PATH_PREFIX}%")
    ).all()
    for resume in seeded:
        db.session.delete(resume)
    db.session.commit()
    return len(seeded)


def seed_resumes(limit: int = 50, per_category: int = 3, reset: bool = False) -> dict:
    """
    Insert sample resumes from the corpus into the database.

    Args:
        limit: Maximum resumes to insert.
        per_category: Resumes to take from each job category, for coverage.
        reset: Remove previously seeded resumes first.

    Returns:
        A summary dict with counts and the seed user's id.
    """
    from app import create_app
    from app.extensions import db
    import app.models as models
    from app.resume.services import ResumeParserService

    from scripts.datasets.loaders import load_resumes

    frame = load_resumes()
    # Seed through the app's own extractor so seeded skills are exactly what a
    # real upload of the same text would produce, boundary matching included.
    vocabulary = vocabulary_terms()

    application = create_app()
    with application.app_context():
        removed = reset_seeded(db, models) if reset else 0
        user = _get_seed_user(db, models)

        # `limit` caps the total seeded population, not the rows added by this
        # run, so re-running without --reset converges instead of topping up.
        already_seeded = models.Resume.query.filter(
            models.Resume.file_path.like(f"{SEED_PATH_PREFIX}%")
        ).count()
        budget = max(limit - already_seeded, 0)

        inserted = 0
        skills_added = 0
        by_category: dict[str, int] = {}

        for category in sorted(frame["label"].unique()):
            if inserted >= budget:
                break
            subset = frame[frame["label"] == category].head(per_category)
            for position, (_, row) in enumerate(subset.iterrows()):
                if inserted >= budget:
                    break
                text = str(row["text"])
                marker = f"{SEED_PATH_PREFIX}{row['source']}/{category}/{position}"
                if models.Resume.query.filter_by(file_path=marker).first():
                    continue

                resume = models.Resume(
                    user_id=user.id,
                    file_name=f"{category.lower().replace(' ', '_')}_{position}.txt",
                    file_path=marker,
                    raw_text=text[:20000],
                    name=f"{category} candidate {position + 1}",
                    summary=text[:400],
                    is_primary=False,
                )
                db.session.add(resume)
                db.session.flush()

                matched = ResumeParserService.extract_skills(text)[:25]
                for term in matched:
                    db.session.add(
                        models.Skill(
                            resume_id=resume.id,
                            name=term,
                            category=_skill_category(term),
                            proficiency="Detected",
                        )
                    )
                skills_added += len(matched)
                inserted += 1
                by_category[category] = by_category.get(category, 0) + 1

        db.session.commit()

        totals = {
            "resumes_total": models.Resume.query.count(),
            "skills_total": models.Skill.query.count(),
        }

    return {
        "removed": removed,
        "already_seeded": already_seeded,
        "resumes_inserted": inserted,
        "skills_inserted": skills_added,
        "categories": by_category,
        "seed_user": SEED_USER_EMAIL,
        "database_totals": totals,
    }


def database_snapshot() -> dict:
    """
    Report current row counts for the tables this tooling touches.

    Returns:
        A mapping of table name to row count.
    """
    from app import create_app
    from app.extensions import db
    import app.models as models

    application = create_app()
    with application.app_context():
        seeded = models.Resume.query.filter(
            models.Resume.file_path.like(f"{SEED_PATH_PREFIX}%")
        ).count()
        return {
            "users": models.User.query.count(),
            "resumes": models.Resume.query.count(),
            "resumes_seeded": seeded,
            "skills": models.Skill.query.count(),
            "interviews": models.Interview.query.count(),
            "questions": models.Question.query.count(),
            "database": str(
                db.engine.url if db.engine is not None else "unknown"
            ),
        }
