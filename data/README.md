# Datasets

Kaggle NLP corpora used by InterviewIQ / SmartHireAI, plus everything derived
from them. Nothing in `raw/`, `processed/`, `models/` or `reports/` is committed
— the folders are placeholders and `data/.gitignore` keeps the payloads out of
version control. Re-create them with the commands below.

## Layout

| Folder       | Holds                                                        |
|--------------|--------------------------------------------------------------|
| `raw/`       | Unzipped Kaggle downloads, one folder per dataset key         |
| `processed/` | Normalised CSVs and `skills_vocab.json`                       |
| `models/`    | Trained scikit-learn pipelines (`*.joblib`)                   |
| `reports/`   | Training metrics and evaluation reports (`*.json`)            |

## One-time setup

1. Install the tooling (separate from the app's `requirements.txt`):

   ```
   venv\Scripts\python.exe -m pip install -r requirements-data.txt
   ```

2. Get a Kaggle API token. Either scheme works:

   - **Current** - *Generate New Token* under *API Tokens* at
     <https://www.kaggle.com/settings>. Save the `KGAT_...` string to
     `~/.kaggle/access_token`, or export `KAGGLE_API_TOKEN`. Requires
     `kaggle>=1.8.0`; older clients do not understand this scheme.
   - **Legacy** - *Create New Token* in the older *API* section downloads
     `kaggle.json`. Save it to `~/.kaggle/kaggle.json`, or export
     `KAGGLE_USERNAME` and `KAGGLE_KEY`.

   Treat the token as a password. If it is ever shown in a screenshot, a chat
   or a commit, expire it and generate a new one.

3. Confirm the slugs still resolve before downloading anything:

   ```
   venv\Scripts\python.exe -m scripts.datasets.cli verify
   ```

## Registered datasets

| Key               | Kaggle slug                            | Task                  | Used for |
|-------------------|----------------------------------------|-----------------------|----------|
| `resumes_large`   | `snehaanbhawal/resume-dataset`         | resume classification | Resume category model; source corpus for the mined skill vocabulary and the seeded sample resumes |
| `resumes_classic` | `jillanisofttech/updated-resume-dataset` | resume classification | IT-role-labelled resumes. Feeds skill mining and seeding, not the classifier |
| `interview_questions` | `syedmharis/software-engineering-interview-questions-dataset` | interview questions | 174 questions across 22 topics. Downloaded but not yet wired in |
| `emotions_text`   | `praveengovi/emotions-dataset-for-nlp` | text emotion          | Sentence-level emotion model over answer transcripts |
| `sentiment140`    | `kazanova/sentiment140`                | sentiment             | Optional. Large polarity baseline; not used for seeding because tweets are stylistically far from interview answers |

All slugs were verified live on 2026-09-14. `gauravduttakiit/resume-dataset`,
the original `resumes_classic` source, now returns 403 and was replaced with the
`jillanisofttech` mirror of the same file.

Slugs are declared in `scripts/datasets/manifest.py` and are **not** verified at
import time. Kaggle contributors rename and delete datasets, so `verify` is the
command that tells you whether a slug is still live. If one has moved:

```
venv\Scripts\python.exe -m scripts.datasets.cli search "resume dataset"
```

then update `DATASETS` in the manifest.

## Usage

```
# what is registered, downloaded, trained and seeded
python -m scripts.datasets.cli list
python -m scripts.datasets.cli status

# fetch and look at the data
python -m scripts.datasets.cli download
python -m scripts.datasets.cli inspect

# derive things from it
python -m scripts.datasets.cli mine-skills
python -m scripts.datasets.cli train
python -m scripts.datasets.cli evaluate
python -m scripts.datasets.cli seed --limit 50

# all of the above, in order
python -m scripts.datasets.cli all
```

Run these from the `SmartHireAI` project root using `venv\Scripts\python.exe` —
that virtualenv has both Flask (needed by `seed`) and pandas/scikit-learn.

## What each derived artefact is

**`processed/skills_vocab.json`** — skill terms mined from the resume corpus by
per-category TF-IDF. Entries carry a `status`:

- `confirmed` — already present in `ResumeParserService.COMMON_SKILLS` (58 terms)
- `candidate` — newly mined and **unreviewed** (~2,660 terms)

Terms are ranked by *discriminativeness*: a term's mean TF-IDF inside its
category divided by its mean across the whole corpus. Ranking by raw TF-IDF
instead surfaces boilerplate — the first version of this returned "skills",
"management", "new", "professional". The ratio pushes those to ~1.0 and they
drop out, leaving `informatica`, `hdfs`, `sqoop`, `mapreduce`, `cloudera`,
`hbase`. N-gram noise such as "developer dot" still gets through, which is
exactly why these need review.

`ResumeParserService.skill_vocabulary()` matches `confirmed` and `approved`
terms only. Term mining is noisy and surfaces non-skills, so promote a term by
editing its `status` to `approved` after reading it. To match everything
including unreviewed terms, set `SKILL_VOCAB_INCLUDE_CANDIDATES=1`. If the file
is missing or malformed the app silently falls back to `COMMON_SKILLS`, so a
fresh checkout with no datasets behaves exactly as before.

**`models/resume_category.joblib`** — TF-IDF + logistic regression, resume text
to job category. **67.8% accuracy, macro-F1 0.632 across 24 classes** on a
497-row held-out split. It never predicts `BPO`.

Trained on `resumes_large` only. The two resume corpora label at different
granularities — `resumes_large` by industry (`INFORMATION-TECHNOLOGY`,
`ACCOUNTANT`), `resumes_classic` by IT role (`Java Developer`, `Data Science`)
— so concatenating them produces a 45-class union of two vocabularies and a
worse model (64.0%, macro-F1 0.624). `RESUME_TRAIN_SOURCES` in `train.py` pins
this. `resumes_classic` also dedupes from 962 rows to 166 unique resumes, too
thin to fit on its own.

**`models/text_emotion.joblib`** — TF-IDF + logistic regression, sentence to
emotion. **90.4% accuracy, macro-F1 0.874 across 6 classes** on a 3,941-row
held-out split.

`scripts.datasets.evaluate.emotion_scores(text)` folds its six labels
into the four floats `EmotionAnalysis` stores and returns them on a 0-100 scale:

```python
>>> emotion_scores("i feel anxious about this question")
{'happy_score': 1.85, 'neutral_score': 0.93, 'sad_score': 0.92,
 'nervous_score': 96.3, 'dominant_emotion': 'Nervous'}
```

The six-to-four mapping is `EMOTION_TO_APP` in `scripts/datasets/train.py`. It
is an approximation — `anger` and `fear` both collapse onto `Nervous` — and is
kept in one place so it stays reviewable.

Nothing in the app calls `emotion_scores` yet. `app/ai/emotion_service.py` is
purely visual (OpenCV on video frames) and never reads the transcript, so this
is a complementary text-side signal, not a replacement.

## Seeded database rows

`seed` inserts sample resumes and their skills under a dedicated account,
`dataset.seed@interviewiq.local`, which has no password hash and no `google_id`
and therefore cannot be signed into. Seeded resumes:

- carry a `dataset://` prefix in `file_path`, which is how `--reset` finds them
- are never `is_primary`, so a real candidate upload still wins
- count against `--limit` on re-runs, so seeding twice does not duplicate rows

```
python -m scripts.datasets.cli seed --limit 50 --reset   # rebuild from scratch
python -m scripts.datasets.cli seed --limit 0  --reset   # remove all seeded rows
```

## Skill matching is boundary-anchored

`extract_skills` matches on `(?<![a-z0-9])term(?![a-z0-9])`, not plain substring.
Substring matching makes short terms catastrophic - `Go` matches inside *going*,
*good* and *category*. Lookarounds are used rather than `\b` because `\b` does
not behave at the edges of `C++`, `.NET` or `Node.js`.

Seeding calls `ResumeParserService.extract_skills` directly, so seeded skills are
exactly what a real upload of the same text would produce. With the naive
matcher, seeding 48 resumes produced 1,200 skill rows led by "skills", "go",
"service", "new"; it now produces 117 led by Communication, Leadership, SQL,
HTML, Python.

## Why questions are not seeded

`questions.interview_id` is `NOT NULL`, so a question row only exists inside an
interview session — there is no standalone question-bank table to seed.
Question generation stays with `GeminiService.generate_interview_questions`, backed by the
hard-coded `_fallback_questions` when the API is unavailable.

Seeding a real question bank needs a new table. The corpus now exists -
`interview_questions` is downloaded, 174 questions across 22 topics
(System Design 24, DevOps 15, Security 13, Data Structures 11, ...) - but nothing
reads it yet. Wiring it up means adding a `QuestionBank` model and a seeder,
which `db.create_all()` would pick up automatically.

## Known wrinkle

Installing `requirements-data.txt` pulls newer `urllib3` and `chardet` than
`requests 2.32.3` advertises support for, so `import requests` prints a
`RequestsDependencyWarning`. It is cosmetic — `requests` was verified working
after the install — but `-W ignore` silences it if the noise is annoying.
