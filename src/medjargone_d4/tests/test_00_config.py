"""
Level 0 — Config and imports.
No external APIs, no models, no network.

Run: python -m medjargone.tests.test_00_config
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medjargone_d4 import config

PASS = "  PASS"
FAIL = "  FAIL"


def test_project_root():
    assert config.PROJECT_ROOT.exists(), f"PROJECT_ROOT not found: {config.PROJECT_ROOT}"
    print(PASS, f"PROJECT_ROOT = {config.PROJECT_ROOT}")


def test_data_dirs():
    # These don't need to exist yet — just verify the paths are sensible
    assert "medjargone" in str(config.DATA_DIR)
    assert "umls_subset.sqlite" in str(config.UMLS_SUBSET_DB)
    print(PASS, "Data paths look correct")


def test_umls_api_key():
    key = config.UMLS_API_KEY
    if key:
        print(PASS, f"UMLS_API_KEY is set ({len(key)} chars)")
    else:
        print("  WARN  UMLS_API_KEY is NOT set — API tests will be skipped")
        print("        export UMLS_API_KEY=<your_key>")


def test_local_umls_subset():
    if config.UMLS_SUBSET_DB.exists():
        size_mb = config.UMLS_SUBSET_DB.stat().st_size / 1e6
        print(PASS, f"Local UMLS subset found ({size_mb:.1f} MB)")
    else:
        print("  WARN  umls_subset.sqlite not built yet — offline glossary disabled")
        print("        Run: python -m medjargone.setup.build_umls_index")


def test_curated_anatomy():
    import json
    assert config.CURATED_ANATOMY.exists(), f"curated_anatomy.json missing: {config.CURATED_ANATOMY}"
    data = json.loads(config.CURATED_ANATOMY.read_text())
    assert len(data) > 10
    print(PASS, f"curated_anatomy.json loaded ({len(data)} entries)")


def test_test_data():
    test_dir = config.MULTICLINSUM_TEST_DIR / "fulltext"
    files = list(test_dir.glob("*.txt")) if test_dir.exists() else []
    assert files, f"No test files found at {test_dir}"
    print(PASS, f"Test data found ({len(files)} files at {test_dir})")


if __name__ == "__main__":
    print("=== Level 0: Config checks ===")
    for fn in [test_project_root, test_data_dirs, test_umls_api_key,
               test_local_umls_subset, test_curated_anatomy, test_test_data]:
        try:
            fn()
        except Exception as e:
            print(FAIL, fn.__name__, "—", e)
    print()
