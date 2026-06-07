"""
Download a UMLS release via the UTS download endpoint and run MetamorphoSys to
build a SAB-restricted subset.

Usage:
    python -m medjargone.setup.download_umls --list-releases
    python -m medjargone.setup.download_umls --release 2024AB --dest data/medjargone/umls_raw/

After download, run build_umls_index.py to load the subset into SQLite.

Note: This script downloads multi-GB files and invokes MetamorphoSys (a Java GUI
or batch tool bundled with the release). For CI/headless use the --batch flag runs
MetamorphoSys with a pre-configured properties file that restricts SABs to:
  CHV, SNOMEDCT_US, RXNORM, MSH, NCI, MED-RT, MTHSPL

UMLS-derived data must NOT be redistributed. Keep the subset local.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medjargone_d4 import config

RELEASES_URL = "https://uts-ws.nlm.nih.gov/releases"
DOWNLOAD_BASE = "https://uts-ws.nlm.nih.gov/download"

# Minimal MetamorphoSys batch config — restricts SABs and keeps ENG only.
_MMSYS_CONFIG_TEMPLATE = """\
gov.nih.nlm.umls.meta.config.MetamorphoSysConfig.release_version={release}
gov.nih.nlm.umls.mmsys.filter.SourceListFilter.selected_sources=\\
  CHV|SNOMEDCT_US|RXNORM|MSH|NCI|MED-RT|MTHSPL
gov.nih.nlm.umls.meta.config.MetamorphoSysConfig.language_list=ENG
gov.nih.nlm.umls.meta.config.MetamorphoSysConfig.suppress_pref=N
gov.nih.nlm.umls.meta.config.MetamorphoSysConfig.output_type=TEXT
gov.nih.nlm.umls.meta.config.MetamorphoSysConfig.output_dir={output_dir}
"""


def _get(url: str, params: dict) -> dict:
    import requests
    r = requests.get(url, params=params, timeout=config.API_TIMEOUT)
    r.raise_for_status()
    return r.json()


def list_releases(api_key: str) -> None:
    data = _get(RELEASES_URL, {"apiKey": api_key})
    for rel in data.get("result", []):
        print(rel.get("releaseVersion"), rel.get("releaseDate"), rel.get("downloadUrl", ""))


def download_release(api_key: str, release: str, dest: Path) -> Path:
    """Download the full UMLS release zip to dest/. Returns the zip path."""
    import requests
    dest.mkdir(parents=True, exist_ok=True)

    # Locate the download URL for this release
    releases = _get(RELEASES_URL, {"apiKey": api_key}).get("result", [])
    release_info = next((r for r in releases if r.get("releaseVersion") == release), None)
    if not release_info:
        print(f"Release '{release}' not found. Use --list-releases to see available.")
        sys.exit(1)

    download_url = release_info["downloadUrl"]
    zip_path = dest / f"umls-{release}-full.zip"

    if zip_path.exists():
        print(f"Already downloaded: {zip_path}")
        return zip_path

    print(f"Downloading UMLS {release} (~3–5 GB) → {zip_path} …")
    with requests.get(
        DOWNLOAD_BASE, params={"url": download_url, "apiKey": api_key},
        stream=True, timeout=300
    ) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        written = 0
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                written += len(chunk)
                if total:
                    pct = written / total * 100
                    print(f"\r  {pct:.1f}%  {written >> 20} MB / {total >> 20} MB", end="")
    print(f"\nDownload complete: {zip_path}")
    return zip_path


def run_mmsys_batch(release: str, release_dir: Path, output_dir: Path) -> None:
    """Run MetamorphoSys in batch mode with the SAB-restricted config."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_text = _MMSYS_CONFIG_TEMPLATE.format(
        release=release,
        output_dir=str(output_dir),
    )
    config_path = release_dir / "mmsys_batch.prop"
    config_path.write_text(config_text)

    # MetamorphoSys ships as a shell script inside the release zip
    mmsys_sh = release_dir / "mmsys.sh"
    if not mmsys_sh.exists():
        print(f"[warn] mmsys.sh not found at {mmsys_sh}")
        print("Unzip the release first, then re-run with --release-dir.")
        return

    print(f"Running MetamorphoSys batch subset extraction → {output_dir} …")
    subprocess.run(
        ["bash", str(mmsys_sh), "-batch", str(config_path)],
        check=True,
    )
    print("MetamorphoSys complete. Run build_umls_index.py next.")


def main():
    parser = argparse.ArgumentParser(description="Download UMLS release via UTS")
    parser.add_argument("--list-releases", action="store_true")
    parser.add_argument("--release",     default="2024AB", help="UMLS release version")
    parser.add_argument("--dest",        type=Path, default=Path("data/medjargone/umls_raw"))
    parser.add_argument("--release-dir", type=Path, default=None,
                        help="Unzipped release dir (skip download if set)")
    parser.add_argument("--output-dir",  type=Path, default=Path("data/medjargone/umls_subset_rrf"),
                        help="Where MetamorphoSys writes the RRF subset")
    args = parser.parse_args()

    api_key = config.UMLS_API_KEY
    if not api_key:
        print("Error: UMLS_API_KEY environment variable is not set.")
        print("  export UMLS_API_KEY=your_key_from_uts_profile")
        sys.exit(1)

    if args.list_releases:
        list_releases(api_key)
        return

    if args.release_dir is None:
        zip_path = download_release(api_key, args.release, args.dest)
        release_dir = args.dest / f"umls-{args.release}-full"
        if not release_dir.exists():
            print(f"Unzip the release: unzip {zip_path} -d {release_dir}")
            print("Then re-run with --release-dir to proceed to MetamorphoSys.")
            return
    else:
        release_dir = args.release_dir

    run_mmsys_batch(args.release, release_dir, args.output_dir)


if __name__ == "__main__":
    main()
