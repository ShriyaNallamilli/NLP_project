"""
Kaggle NLP dataset tooling for InterviewIQ / SmartHireAI.

Sub-modules:
    manifest        Registry of the Kaggle datasets this project uses.
    kaggle_client   Authentication, verification and download helpers.
    loaders         Normalise each raw dataset into a canonical DataFrame.
    seed_db         Seed smarthire.db and the skill vocabulary from the corpora.
    train           Train TF-IDF text classifiers on the corpora.
    evaluate        Score the trained classifiers on a held-out split.
    cli             Command line entry point (`python -m scripts.datasets.cli`).
"""
