"""
Command line entry point for the dataset tooling.

Run from the SmartHireAI project root so ``scripts`` is importable:

    venv\\Scripts\\python.exe -m scripts.datasets.cli list
    venv\\Scripts\\python.exe -m scripts.datasets.cli verify
    venv\\Scripts\\python.exe -m scripts.datasets.cli download
    venv\\Scripts\\python.exe -m scripts.datasets.cli all

``list``, ``status`` and ``--help`` work with no Kaggle token; everything that
touches the network states plainly what it needs.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Allow `python scripts/datasets/cli.py` as well as `-m scripts.datasets.cli`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.datasets import kaggle_client, manifest


def _print_rows(rows: list[dict], columns: list[tuple[str, int]]) -> None:
    """Render dicts as a fixed-width table."""
    header = "".join(name.ljust(width) for name, width in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        line = ""
        for name, width in columns:
            value = str(row.get(name, ""))
            if len(value) > width - 2:
                value = value[: width - 3] + "…"
            line += value.ljust(width)
        print(line)


def cmd_list(args: argparse.Namespace) -> int:
    """Print the dataset registry without touching the network."""
    print(f"Registered NLP datasets ({len(manifest.DATASETS)}):\n")
    for spec in manifest.DATASETS:
        downloaded = spec.raw_dir.is_dir() and any(
            p.is_file() and p.name != ".gitkeep" for p in spec.raw_dir.glob("**/*")
        )
        flags = []
        if spec.optional:
            flags.append("optional")
        flags.append("downloaded" if downloaded else "not downloaded")
        print(f"  {spec.key}  [{', '.join(flags)}]")
        print(f"    slug  : {spec.slug}")
        print(f"    title : {spec.title}")
        print(f"    task  : {spec.task}")
        print(f"    use   : {spec.why}")
        print()
    print(f"Raw data goes to: {manifest.RAW_DIR}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Check that every selected slug resolves on Kaggle."""
    specs = manifest.select(args.datasets, include_optional=True)
    api = kaggle_client.get_api()
    rows = []
    failures = 0
    for spec in specs:
        result = kaggle_client.verify(spec, api=api)
        if result["ok"]:
            size_mb = result["total_bytes"] / (1024 * 1024)
            rows.append(
                {
                    "key": spec.key,
                    "slug": spec.slug,
                    "status": "ok",
                    "detail": f"{len(result['files'])} files, {size_mb:.1f} MB",
                }
            )
        else:
            failures += 1
            rows.append(
                {
                    "key": spec.key,
                    "slug": spec.slug,
                    "status": "FAILED",
                    "detail": result["error"],
                }
            )
    _print_rows(rows, [("key", 20), ("slug", 42), ("status", 10), ("detail", 60)])
    if failures:
        print(
            f"\n{failures} slug(s) did not resolve. Find a replacement with:\n"
            f'  python -m scripts.datasets.cli search "resume dataset"\n'
            f"then update DATASETS in scripts/datasets/manifest.py."
        )
    return 1 if failures else 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search Kaggle for datasets by keyword."""
    rows = kaggle_client.search(args.term, limit=args.limit)
    if not rows:
        print(f"No datasets matched '{args.term}'.")
        return 1
    _print_rows(rows, [("ref", 48), ("title", 52), ("size", 12), ("downloads", 12)])
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download the selected datasets into data/raw."""
    specs = manifest.select(args.datasets, include_optional=args.include_optional)
    api = kaggle_client.get_api()
    results = []
    for spec in specs:
        print(f"\n>> {spec.key} ({spec.slug})")
        result = kaggle_client.download(spec, force=args.force, api=api)
        status = result["status"]
        if status == "failed":
            print(f"   FAILED: {result.get('error')}")
        else:
            print(f"   {status}: {len(result.get('files', []))} file(s) in {result['path']}")
        results.append(result)

    destination = kaggle_client.write_download_manifest(
        results, manifest.DATA_DIR / "download_manifest.json"
    )
    print(f"\nManifest written to {destination}")
    return 1 if any(r["status"] == "failed" for r in results) else 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Show shape and label distribution for each downloaded dataset."""
    from scripts.datasets.loaders import DatasetNotDownloaded, load_dataset

    exit_code = 0
    for spec in manifest.select(args.datasets, include_optional=True):
        print(f"\n=== {spec.key} ({spec.task}) ===")
        try:
            frame = load_dataset(spec, max_rows=args.max_rows)
        except (DatasetNotDownloaded, ValueError) as exc:
            print(f"  unavailable: {exc}")
            exit_code = 1
            continue
        print(f"  rows: {len(frame)}   labels: {frame['label'].nunique()}")
        print(f"  mean text length: {frame['text'].str.len().mean():.0f} chars")
        print("  top labels:")
        for label, count in frame["label"].value_counts().head(10).items():
            print(f"    {str(label)[:40]:<42}{count}")
    return exit_code


def cmd_mine_skills(args: argparse.Namespace) -> int:
    """Mine a skill vocabulary from the resume corpus."""
    from scripts.datasets.train import mine_skill_vocabulary

    vocabulary = mine_skill_vocabulary(
        top_per_category=args.top_per_category, min_documents=args.min_documents
    )
    print(
        f"Mined {vocabulary['mined_terms']} candidate terms "
        f"from {vocabulary['corpus_rows']} resumes "
        f"(plus {vocabulary['seed_terms']} confirmed seed terms)."
    )
    print(f"Written to {vocabulary['path']}")
    print(
        "\nMined terms are marked 'candidate' and are noisy by nature - review "
        "them before trusting them in scoring."
    )
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the selected classifiers."""
    from scripts.datasets.loaders import DatasetNotDownloaded
    from scripts.datasets.train import train_emotion_classifier, train_resume_classifier

    jobs = {"resume": train_resume_classifier, "emotion": train_emotion_classifier}
    selected = args.models or list(jobs)
    exit_code = 0

    for name in selected:
        print(f"\n>> training {name}")
        try:
            metrics = jobs[name](max_rows=args.max_rows)
        except (DatasetNotDownloaded, ValueError) as exc:
            print(f"   skipped: {exc}")
            exit_code = 1
            continue
        split = metrics["split"]
        print(
            f"   rows={metrics['rows_total']} classes={len(metrics['classes'])} "
            f"(train {split['train']} / val {split['val']} / test {split['test']})"
        )
        print(
            f"   val  accuracy={metrics['val_accuracy']:.1%} "
            f"f1_macro={metrics['val_f1_macro']:.3f}   <- compare models on this"
        )
        print(
            f"   test accuracy={metrics['accuracy']:.1%} "
            f"f1_macro={metrics['f1_macro']:.3f}   <- report this once"
        )
        if metrics.get("refit_on"):
            print(f"   shipped model refit on {metrics['refit_on']} ({metrics['refit_rows']} rows)")
        print(f"   model  -> {metrics['model_path']}")
        print(f"   report -> {metrics['report_path']}")
    return exit_code


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate the trained classifiers on their held-out splits."""
    from scripts.datasets.evaluate import (
        evaluate_emotion_classifier,
        evaluate_resume_classifier,
        summarise,
    )
    from scripts.datasets.loaders import DatasetNotDownloaded

    jobs = {"resume": evaluate_resume_classifier, "emotion": evaluate_emotion_classifier}
    selected = args.models or list(jobs)
    reports = []
    exit_code = 0

    for name in selected:
        try:
            reports.append(jobs[name]())
        except (DatasetNotDownloaded, FileNotFoundError, ValueError) as exc:
            print(f"{name}: skipped - {exc}")
            exit_code = 1

    if reports:
        print()
        print(summarise(reports))
        for report in reports:
            print(f"\n{report['model']} report -> {report['report_path']}")
    return exit_code


def _corpora(selected: list[str] | None) -> dict:
    """
    Load the corpora the analysis commands operate on.

    Args:
        selected: ``resume`` and/or ``emotion``, or None for both.

    Returns:
        Mapping of model name to (corpus frame, vectoriser vocabulary cap).
    """
    from scripts.datasets.loaders import load_resumes, load_text_emotions
    from scripts.datasets.train import RESUME_TRAIN_SOURCES

    wanted = selected or ["resume", "emotion"]
    corpora: dict[str, tuple] = {}

    if "resume" in wanted:
        corpora["resume_category"] = (
            load_resumes(sources=RESUME_TRAIN_SOURCES),
            60_000,
        )
    if "emotion" in wanted:
        frame = load_text_emotions()
        frame["label"] = frame["label"].str.lower().str.strip()
        corpora["text_emotion"] = (frame, 40_000)
    return corpora


def cmd_baselines(args: argparse.Namespace) -> int:
    """Compare trivial and intermediate baselines against the shipped model."""
    from scripts.datasets.baselines import format_table, lift_over_floor, run_baselines
    from scripts.datasets.loaders import DatasetNotDownloaded
    from scripts.datasets.splits import make_split

    exit_code = 0
    for name, (frame, max_features) in _corpora(args.models).items():
        print(f"\n{'=' * 74}\n== {name}\n{'=' * 74}")
        try:
            split = make_split(frame)
        except (DatasetNotDownloaded, ValueError) as exc:
            print(f"  skipped: {exc}")
            exit_code = 1
            continue

        report = run_baselines(
            split,
            name,
            max_features=max_features,
            cross_validate=not args.no_cv,
        )
        print(format_table(report))
        print(f"\n  {lift_over_floor(report)}")
        print(f"  report -> {report['report_path']}")
    return exit_code


def cmd_errors(args: argparse.Namespace) -> int:
    """Explain what the trained models get wrong, and why."""
    from scripts.datasets.error_analysis import analyse, format_analysis
    from scripts.datasets.loaders import DatasetNotDownloaded
    from scripts.datasets.splits import make_split
    from scripts.datasets.train import load_model

    exit_code = 0
    for name, (frame, _) in _corpora(args.models).items():
        print(f"\n{'=' * 74}\n== {name}\n{'=' * 74}")
        try:
            split = make_split(frame)
            pipeline = load_model(name)
        except (DatasetNotDownloaded, FileNotFoundError, ValueError) as exc:
            print(f"  skipped: {exc}")
            exit_code = 1
            continue

        analysis = analyse(
            name,
            pipeline,
            split.test,
            split.train["label"].value_counts(),
            fold="test",
        )
        print(format_analysis(analysis))
        print(f"\n  report -> {analysis['report_path']}")
    return exit_code


def cmd_embeddings(args: argparse.Namespace) -> int:
    """Compare frozen transformer embeddings against the TF-IDF representation."""
    import json as _json

    from scripts.datasets.embeddings import EncoderUnavailable, compare, format_report
    from scripts.datasets.loaders import DatasetNotDownloaded
    from scripts.datasets.manifest import REPORT_DIR

    exit_code = 0
    for name, (frame, _) in _corpora(args.models).items():
        print(f"\n{'=' * 74}\n== {name}\n{'=' * 74}")

        # Compare against the *best* sparse baseline, not an arbitrary one.
        # Beating a deliberately weak reference would prove nothing.
        tfidf_accuracy = None
        tfidf_name = None
        baseline_report = REPORT_DIR / f"{name}_baselines.json"
        if baseline_report.is_file():
            results = _json.loads(baseline_report.read_text(encoding="utf-8"))["results"]
            sparse = [
                r
                for r in results
                if r["model"] not in {"majority_class", "stratified_random"}
            ]
            if sparse:
                best = max(sparse, key=lambda r: r["val_accuracy"])
                tfidf_accuracy = best["val_accuracy"]
                tfidf_name = best["model"]

        try:
            report = compare(
                frame, name, tfidf_val_accuracy=tfidf_accuracy, tfidf_name=tfidf_name
            )
        except (EncoderUnavailable, DatasetNotDownloaded, ValueError) as exc:
            print(f"  skipped: {exc}")
            exit_code = 1
            continue

        print(format_report(report))
        print(f"\n  report -> {report['report_path']}")
    return exit_code


def cmd_report(args: argparse.Namespace) -> int:
    """Generate the written evaluation report from the measured artefacts."""
    from scripts.datasets.report import generate

    destination = generate()
    print(f"Report written to {destination}")
    print(
        "\nSections with no data say so explicitly - run baselines, errors and "
        "embeddings first for a complete report."
    )
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    """Seed sample resumes and skills into smarthire.db."""
    from scripts.datasets.seed_db import seed_resumes

    summary = seed_resumes(
        limit=args.limit, per_category=args.per_category, reset=args.reset
    )
    if summary["removed"]:
        print(f"Removed {summary['removed']} previously seeded resume(s).")
    if summary["already_seeded"]:
        print(
            f"{summary['already_seeded']} seeded resume(s) already present "
            f"and counted against --limit."
        )
    print(
        f"Inserted {summary['resumes_inserted']} resume(s) and "
        f"{summary['skills_inserted']} skill row(s) "
        f"under {summary['seed_user']}."
    )
    print(f"Database now holds: {json.dumps(summary['database_totals'])}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Report what is downloaded, trained and seeded."""
    print("Kaggle credentials:", kaggle_client.credentials_kind())
    print("  API token :", kaggle_client.access_token_path())
    print("  legacy    :", kaggle_client.credentials_path())

    print("\nDownloads:")
    for spec in manifest.DATASETS:
        files = (
            [p for p in spec.raw_dir.glob("**/*") if p.is_file() and p.name != ".gitkeep"]
            if spec.raw_dir.is_dir()
            else []
        )
        size_mb = sum(p.stat().st_size for p in files) / (1024 * 1024)
        state = f"{len(files)} files, {size_mb:.1f} MB" if files else "not downloaded"
        print(f"  {spec.key:<20}{state}")

    print("\nModels:")
    for name in ("resume_category", "text_emotion"):
        path = manifest.MODEL_DIR / f"{name}.joblib"
        print(f"  {name:<20}{'trained' if path.is_file() else 'not trained'}")

    vocabulary = manifest.PROCESSED_DIR / "skills_vocab.json"
    print(f"\nSkill vocabulary: {'built' if vocabulary.is_file() else 'not built'}")

    print("\nDatabase:")
    try:
        from scripts.datasets.seed_db import database_snapshot

        for key, value in database_snapshot().items():
            print(f"  {key:<20}{value}")
    except Exception as exc:  # noqa: BLE001 - status must never hard-fail
        print(f"  unavailable: {exc}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    """Run download, mine, train, evaluate and seed in order."""
    steps = [
        ("download", cmd_download),
        ("mine-skills", cmd_mine_skills),
        ("train", cmd_train),
        ("evaluate", cmd_evaluate),
        ("seed", cmd_seed),
    ]
    for label, handler in steps:
        print(f"\n{'=' * 70}\n== {label}\n{'=' * 70}")
        code = handler(args)
        if code != 0 and label == "download":
            print("\nDownload failed; stopping before the dependent steps.")
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="scripts.datasets.cli",
        description="Fetch, inspect, train on and seed Kaggle NLP datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_dataset_flag(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--datasets",
            nargs="+",
            metavar="KEY",
            help="Dataset keys to act on (default: all non-optional).",
        )

    listing = subparsers.add_parser("list", help="Show the dataset registry.")
    listing.set_defaults(func=cmd_list)

    verify = subparsers.add_parser("verify", help="Check slugs resolve on Kaggle.")
    add_dataset_flag(verify)
    verify.set_defaults(func=cmd_verify)

    search = subparsers.add_parser("search", help="Search Kaggle for datasets.")
    search.add_argument("term", help='Search text, e.g. "interview questions".')
    search.add_argument("--limit", type=int, default=20)
    search.set_defaults(func=cmd_search)

    download = subparsers.add_parser("download", help="Download datasets.")
    add_dataset_flag(download)
    download.add_argument("--include-optional", action="store_true")
    download.add_argument("--force", action="store_true", help="Re-download.")
    download.set_defaults(func=cmd_download)

    inspect = subparsers.add_parser("inspect", help="Show shapes and label counts.")
    add_dataset_flag(inspect)
    inspect.add_argument("--max-rows", type=int, default=None)
    inspect.set_defaults(func=cmd_inspect)

    mine = subparsers.add_parser("mine-skills", help="Mine the skill vocabulary.")
    mine.add_argument("--top-per-category", type=int, default=60)
    mine.add_argument("--min-documents", type=int, default=5)
    mine.set_defaults(func=cmd_mine_skills)

    train = subparsers.add_parser("train", help="Train the classifiers.")
    train.add_argument("--models", nargs="+", choices=["resume", "emotion"])
    train.add_argument("--max-rows", type=int, default=None)
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate trained models.")
    evaluate.add_argument("--models", nargs="+", choices=["resume", "emotion"])
    evaluate.set_defaults(func=cmd_evaluate)

    baselines = subparsers.add_parser(
        "baselines", help="Compare baselines and run k-fold cross-validation."
    )
    baselines.add_argument("--models", nargs="+", choices=["resume", "emotion"])
    baselines.add_argument(
        "--no-cv", action="store_true", help="Skip cross-validation (much faster)."
    )
    baselines.set_defaults(func=cmd_baselines)

    errors = subparsers.add_parser(
        "errors", help="Explain what the trained models get wrong."
    )
    errors.add_argument("--models", nargs="+", choices=["resume", "emotion"])
    errors.set_defaults(func=cmd_errors)

    embeddings = subparsers.add_parser(
        "embeddings", help="Compare transformer embeddings against TF-IDF."
    )
    embeddings.add_argument("--models", nargs="+", choices=["resume", "emotion"])
    embeddings.set_defaults(func=cmd_embeddings)

    report = subparsers.add_parser(
        "report", help="Write the Markdown evaluation report from measured results."
    )
    report.set_defaults(func=cmd_report)

    seed = subparsers.add_parser("seed", help="Seed resumes and skills into the DB.")
    seed.add_argument("--limit", type=int, default=50)
    seed.add_argument("--per-category", type=int, default=3)
    seed.add_argument("--reset", action="store_true", help="Drop prior seeded rows.")
    seed.set_defaults(func=cmd_seed)

    status = subparsers.add_parser("status", help="Show pipeline state.")
    status.set_defaults(func=cmd_status)

    everything = subparsers.add_parser("all", help="Run the whole pipeline.")
    add_dataset_flag(everything)
    everything.add_argument("--include-optional", action="store_true")
    everything.add_argument("--force", action="store_true")
    everything.add_argument("--models", nargs="+", choices=["resume", "emotion"])
    everything.add_argument("--max-rows", type=int, default=None)
    everything.add_argument("--top-per-category", type=int, default=60)
    everything.add_argument("--min-documents", type=int, default=5)
    everything.add_argument("--limit", type=int, default=50)
    everything.add_argument("--per-category", type=int, default=3)
    everything.add_argument("--reset", action="store_true")
    everything.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Parse arguments and dispatch.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    # Dataset titles routinely contain emoji, which the default cp1252 console
    # on Windows cannot encode; without this, printing a search result raises.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    manifest.ensure_dirs()
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except RuntimeError as exc:
        # Credential and installation problems: message is already actionable.
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - show the trace, do not swallow it
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
