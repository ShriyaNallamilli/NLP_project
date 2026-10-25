"""
Thin wrapper around the Kaggle API.

The ``kaggle`` package authenticates at *import* time and raises if no token is
present, so every import in this module is deliberately lazy and guarded by
:func:`ensure_credentials`. That keeps ``--help``, ``list`` and the offline
loaders usable on a machine with no Kaggle account.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from scripts.datasets.manifest import DatasetSpec, ensure_dirs

CREDENTIALS_HELP = """
Kaggle credentials were not found.

Sign in at https://www.kaggle.com/settings, then use either scheme:

  API token (current)   'Generate New Token' under 'API Tokens'. Save the
                        KGAT_... string to:
                          {access_token}
                        or export KAGGLE_API_TOKEN. Needs kaggle >= 1.8.0.

  Legacy credentials    'Create New Token' in the older 'API' section, which
                        downloads kaggle.json. Save it to:
                          {json_path}
                        or export KAGGLE_USERNAME and KAGGLE_KEY.

Datasets also need to be accepted once in the browser if they carry a licence
prompt; an otherwise valid token returns 403 until you do.
""".strip()


def config_dir() -> Path:
    """Return the directory the Kaggle client reads credentials from."""
    configured = os.environ.get("KAGGLE_CONFIG_DIR")
    return Path(configured) if configured else Path.home() / ".kaggle"


def credentials_path() -> Path:
    """Return the location of the legacy ``kaggle.json`` credentials file."""
    return config_dir() / "kaggle.json"


def access_token_path() -> Path:
    """Return the location of the current-scheme ``access_token`` file."""
    return config_dir() / "access_token"


def has_credentials() -> bool:
    """Report whether credentials of either scheme are available."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return access_token_path().is_file() or credentials_path().is_file()


def credentials_kind() -> str:
    """Name the credential scheme in use, for status output."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "API token (env)"
    if access_token_path().is_file():
        return "API token (file)"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "legacy credentials (env)"
    if credentials_path().is_file():
        return "legacy credentials (file)"
    return "none"


def ensure_credentials() -> None:
    """
    Verify Kaggle credentials are present before importing the client.

    Raises:
        RuntimeError: With setup instructions when no token can be found.
    """
    if not has_credentials():
        raise RuntimeError(
            CREDENTIALS_HELP.format(
                access_token=access_token_path(), json_path=credentials_path()
            )
        )


def get_api() -> Any:
    """
    Build an authenticated Kaggle API client.

    Returns:
        An authenticated ``KaggleApi`` instance.

    Raises:
        RuntimeError: If credentials are missing or the package is not installed.
    """
    ensure_credentials()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is not installed. Run:\n"
            "  python -m pip install -r requirements-data.txt"
        ) from exc

    api = KaggleApi()
    api.authenticate()
    return api


def _unzip_in_place(target: Path) -> None:
    """
    Expand and remove any .zip files in a folder.

    ``dataset_download_file`` returns a zip when the member is large enough,
    and a plain file otherwise, so the result has to be normalised either way.

    Args:
        target: Folder to expand archives in.
    """
    import zipfile

    for archive in list(target.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
        archive.unlink()


def _file_size(entry: Any) -> int:
    """Read a file size off a Kaggle file record across API versions."""
    for attribute in ("totalBytes", "total_bytes", "size"):
        value = getattr(entry, attribute, None)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def verify(spec: DatasetSpec, api: Any | None = None) -> dict:
    """
    Check that a dataset slug resolves and report its files.

    Args:
        spec: The dataset to check.
        api: Optional pre-built client, to avoid re-authenticating in a loop.

    Returns:
        A dict with ``key``, ``slug``, ``ok``, ``files``, ``total_bytes`` and,
        on failure, ``error``.
    """
    client = api or get_api()
    try:
        listing = client.dataset_list_files(spec.slug)
    except Exception as exc:  # noqa: BLE001 - surface any API/network failure
        return {"key": spec.key, "slug": spec.slug, "ok": False, "error": str(exc)}

    error = getattr(listing, "error_message", None)
    if error:
        return {"key": spec.key, "slug": spec.slug, "ok": False, "error": str(error)}

    files = list(getattr(listing, "files", []) or [])
    if not files:
        return {
            "key": spec.key,
            "slug": spec.slug,
            "ok": False,
            "error": "slug resolved but returned no files (private, removed or renamed)",
        }

    return {
        "key": spec.key,
        "slug": spec.slug,
        "ok": True,
        "files": [getattr(f, "name", str(f)) for f in files],
        "total_bytes": sum(_file_size(f) for f in files),
    }


def search(term: str, limit: int = 20, api: Any | None = None) -> list[dict]:
    """
    Search Kaggle for datasets, for filling gaps the registry does not cover.

    Args:
        term: Free-text search, e.g. ``"interview questions"``.
        limit: Maximum rows to return.
        api: Optional pre-built client.

    Returns:
        A list of ``{ref, title, size, downloads}`` dicts.
    """
    client = api or get_api()
    results = client.dataset_list(search=term)
    rows: list[dict] = []
    for item in list(results)[:limit]:
        rows.append(
            {
                "ref": str(getattr(item, "ref", item)),
                "title": str(getattr(item, "title", "")),
                "size": str(getattr(item, "size", "")),
                "downloads": getattr(item, "downloadCount", None),
            }
        )
    return rows


def download(spec: DatasetSpec, force: bool = False, api: Any | None = None) -> dict:
    """
    Download and unzip one dataset into ``data/raw/<key>/``.

    Args:
        spec: The dataset to fetch.
        force: Re-download even if the target folder already holds files.
        api: Optional pre-built client.

    Returns:
        A dict with ``key``, ``slug``, ``status`` (``downloaded`` / ``cached`` /
        ``failed``), ``path`` and the list of extracted ``files``.
    """
    ensure_dirs()
    target = spec.raw_dir
    existing = [p for p in target.glob("**/*") if p.is_file()] if target.exists() else []

    if existing and not force:
        return {
            "key": spec.key,
            "slug": spec.slug,
            "status": "cached",
            "path": str(target),
            "files": sorted(p.name for p in existing),
        }

    if force and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    client = api or get_api()
    try:
        if spec.download_files:
            for member in spec.download_files:
                client.dataset_download_file(
                    spec.slug, member, path=str(target), quiet=False
                )
            _unzip_in_place(target)
        else:
            client.dataset_download_files(
                spec.slug, path=str(target), unzip=True, quiet=False
            )
    except Exception as exc:  # noqa: BLE001 - report, do not abort the batch
        return {
            "key": spec.key,
            "slug": spec.slug,
            "status": "failed",
            "path": str(target),
            "error": str(exc),
        }

    files = sorted(p.name for p in target.glob("**/*") if p.is_file())
    return {
        "key": spec.key,
        "slug": spec.slug,
        "status": "downloaded" if files else "failed",
        "path": str(target),
        "files": files,
        **({} if files else {"error": "archive extracted to nothing"}),
    }


def write_download_manifest(results: list[dict], destination: Path) -> Path:
    """
    Persist download results so later steps can report data provenance.

    Args:
        results: Rows returned by :func:`download`.
        destination: JSON file to write.

    Returns:
        The path written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return destination
