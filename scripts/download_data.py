#!/usr/bin/env python3
"""One-time online step: download the NASA C-MAPSS turbofan dataset
(FD001 subset) into `data/raw/`.

This is the ONLY script in this project allowed to touch the network.
Everything downstream of `data/raw/` -- pipelines, serving, monitoring --
runs entirely offline, which is what makes Stage 2/3 deployment possible.
Run this once during Stage 1 setup; the files it produces are then
version-controlled via DVC (`dvc add data/raw`) so Stage 2/3 machines
never need to re-download anything.

Usage:
    python scripts/download_data.py [--dest data/raw] [--force]
    PDM_CMAPSS_BASE_URL=<other mirror raw-file base> python scripts/download_data.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests
from loguru import logger

# NASA's own PCoE prognostics data repository has changed hosts/URLs
# repeatedly over the years, so it is not used as the default here.
# Instead this defaults to a plain-text GitHub mirror of the same
# publicly-released FD001 files (verified reachable at build time). It
# can be overridden via PDM_CMAPSS_BASE_URL to point at NASA's current
# repository, a Kaggle mirror you've downloaded manually, or an internal
# mirror -- the URL just needs each FD001 file appended to it directly.
DEFAULT_BASE_URL = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master"
BASE_URL_ENV_VAR = "PDM_CMAPSS_BASE_URL"

# Files that make up the FD001 subset.
FD001_FILES = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]

_MANUAL_FALLBACK_MSG = (
    "Obtain the FD001 files ({files}) from another machine with internet "
    "access -- e.g. NASA's Prognostics Center of Excellence Data Set "
    "Repository, or the Kaggle dataset \"behrad3d/nasa-cmaps\" -- and place "
    "them directly in {dest}. This is the only online step in the project; "
    "everything downstream works offline from these files."
)


def _already_downloaded(dest: Path) -> bool:
    return all((dest / f).exists() for f in FD001_FILES)


def download_and_extract(dest: Path, force: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    if _already_downloaded(dest) and not force:
        logger.info("FD001 files already present in {}; skipping download.", dest)
        return

    base_url = os.environ.get(BASE_URL_ENV_VAR, DEFAULT_BASE_URL)
    logger.info("Downloading C-MAPSS FD001 files from {} ...", base_url)

    for filename in FD001_FILES:
        url = f"{base_url.rstrip('/')}/{filename}"
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(
                "Download of {} failed ({}). " + _MANUAL_FALLBACK_MSG,
                url,
                exc,
                files=", ".join(FD001_FILES),
                dest=dest,
            )
            sys.exit(1)

        target = dest / filename
        target.write_bytes(response.content)
        logger.info("Saved {} ({} bytes)", target, len(response.content))

    if not _already_downloaded(dest):
        logger.error("Download completed but expected FD001 files are missing from {}.", dest)
        sys.exit(1)

    logger.info("C-MAPSS FD001 dataset ready in {}.", dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args()
    download_and_extract(args.dest, force=args.force)


if __name__ == "__main__":
    main()
