"""
M6 — Build the offline UMLS subset SQLite database.

Reads the RRF flat files produced by MetamorphoSys (SAB-restricted subset) and
loads them into umls_subset.sqlite with three tables:

  atoms(cui, str, str_lc, sab, tty, ispref)
      One row per (concept, string, source). Used for term→CUI lookup and for
      extracting CHV consumer-preferred strings (sab='CHV', ispref='Y').

  semtypes(cui, tui, sty)
      Semantic types per concept. Used to filter non-medical concepts.

  definitions(cui, sab, def_text)
      Definitions from MSH / NCI / MedlinePlus. Used as Stage-2 fallback gloss.

Usage:
    # After MetamorphoSys produces a subset at data/medjargone/umls_subset_rrf/:
    python -m medjargone.setup.build_umls_index
    python -m medjargone.setup.build_umls_index --rrf-dir /path/to/rrf/ --output /path/to/db
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medjargone import config

# Default RRF directory (MetamorphoSys output)
DEFAULT_RRF_DIR = config.PROJECT_ROOT / "data" / "medjargone" / "umls_subset_rrf"


# ── Column indices in UMLS RRF files ─────────────────────────────────────────
# MRCONSO.RRF: pipe-separated (no header)
# 0:CUI 1:LAT 2:TS 3:LUI 4:STT 5:SUI 6:ISPREF 7:AUI 8:SAUI 9:SCUI 10:SDUI
# 11:SAB 12:TTY 13:CODE 14:STR 15:SRL 16:SUPPRESS 17:CVF
MRCONSO_CUI    = 0
MRCONSO_LAT    = 1
MRCONSO_ISPREF = 6
MRCONSO_SAB    = 11
MRCONSO_TTY    = 12
MRCONSO_STR    = 14
MRCONSO_SUPP   = 16

# MRSTY.RRF: 0:CUI 1:TUI 2:STN 3:STY 4:ATUI 5:CVF
MRSTY_CUI = 0
MRSTY_TUI = 1
MRSTY_STY = 3

# MRDEF.RRF: 0:CUI 1:AUI 2:ATUI 3:SATUI 4:SAB 5:DEF 6:SUPPRESS 7:CVF
MRDEF_CUI  = 0
MRDEF_SAB  = 4
MRDEF_DEF  = 5
MRDEF_SUPP = 6


def _open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")  # 64 MB page cache
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS atoms;
        DROP TABLE IF EXISTS semtypes;
        DROP TABLE IF EXISTS definitions;

        CREATE TABLE atoms (
            cui     TEXT NOT NULL,
            str     TEXT NOT NULL,
            str_lc  TEXT NOT NULL,
            sab     TEXT NOT NULL,
            tty     TEXT NOT NULL,
            ispref  TEXT NOT NULL
        );

        CREATE TABLE semtypes (
            cui TEXT NOT NULL,
            tui TEXT NOT NULL,
            sty TEXT NOT NULL
        );

        CREATE TABLE definitions (
            cui      TEXT NOT NULL,
            sab      TEXT NOT NULL,
            def_text TEXT NOT NULL
        );
    """)
    conn.commit()


def _build_indices(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_atoms_str_lc ON atoms(str_lc);
        CREATE INDEX IF NOT EXISTS idx_atoms_cui    ON atoms(cui);
        CREATE INDEX IF NOT EXISTS idx_atoms_cui_sab ON atoms(cui, sab);
        CREATE INDEX IF NOT EXISTS idx_semtypes_cui ON semtypes(cui);
        CREATE INDEX IF NOT EXISTS idx_defs_cui      ON definitions(cui);
    """)
    conn.commit()


def load_mrconso(conn: sqlite3.Connection, rrf_dir: Path) -> int:
    filepath = rrf_dir / "MRCONSO.RRF"
    if not filepath.exists():
        print(f"  [error] MRCONSO.RRF not found at {filepath}")
        sys.exit(1)

    rows: list[tuple] = []
    count = 0

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 17:
                continue
            # English only; skip suppressed atoms
            if parts[MRCONSO_LAT] != "ENG":
                continue
            if parts[MRCONSO_SUPP] in ("Y", "E", "O"):
                continue

            cui    = parts[MRCONSO_CUI]
            sab    = parts[MRCONSO_SAB]
            tty    = parts[MRCONSO_TTY]
            ispref = parts[MRCONSO_ISPREF]
            term   = parts[MRCONSO_STR].strip()

            if not term:
                continue

            rows.append((cui, term, term.lower(), sab, tty, ispref))
            count += 1

            if len(rows) >= 50_000:
                conn.executemany(
                    "INSERT INTO atoms VALUES (?,?,?,?,?,?)", rows
                )
                rows.clear()

    if rows:
        conn.executemany("INSERT INTO atoms VALUES (?,?,?,?,?,?)", rows)
    conn.commit()
    return count


def load_mrsty(conn: sqlite3.Connection, rrf_dir: Path) -> int:
    filepath = rrf_dir / "MRSTY.RRF"
    if not filepath.exists():
        print(f"  [error] MRSTY.RRF not found at {filepath}")
        sys.exit(1)

    rows: list[tuple] = []
    count = 0

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 4:
                continue
            cui = parts[MRSTY_CUI]
            tui = parts[MRSTY_TUI]
            sty = parts[MRSTY_STY].strip()
            if cui and sty:
                rows.append((cui, tui, sty))
                count += 1
            if len(rows) >= 50_000:
                conn.executemany("INSERT INTO semtypes VALUES (?,?,?)", rows)
                rows.clear()

    if rows:
        conn.executemany("INSERT INTO semtypes VALUES (?,?,?)", rows)
    conn.commit()
    return count


def load_mrdef(conn: sqlite3.Connection, rrf_dir: Path) -> int:
    filepath = rrf_dir / "MRDEF.RRF"
    if not filepath.exists():
        print(f"  [warn] MRDEF.RRF not found at {filepath} — definitions skipped")
        return 0

    rows: list[tuple] = []
    count = 0

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 6:
                continue
            if parts[MRDEF_SUPP] in ("Y", "E", "O"):
                continue
            cui      = parts[MRDEF_CUI]
            sab      = parts[MRDEF_SAB]
            def_text = parts[MRDEF_DEF].strip()
            if cui and def_text:
                rows.append((cui, sab, def_text))
                count += 1
            if len(rows) >= 50_000:
                conn.executemany("INSERT INTO definitions VALUES (?,?,?)", rows)
                rows.clear()

    if rows:
        conn.executemany("INSERT INTO definitions VALUES (?,?,?)", rows)
    conn.commit()
    return count


def main():
    parser = argparse.ArgumentParser(description="Build UMLS subset SQLite database")
    parser.add_argument(
        "--rrf-dir", type=Path, default=DEFAULT_RRF_DIR,
        help="Directory containing MRCONSO.RRF, MRSTY.RRF, MRDEF.RRF"
    )
    parser.add_argument(
        "--output", type=Path, default=config.UMLS_SUBSET_DB,
        help="Output SQLite path"
    )
    args = parser.parse_args()

    print(f"Building UMLS index from: {args.rrf_dir}")
    print(f"Output: {args.output}")

    conn = _open_db(args.output)
    _create_schema(conn)

    print("Loading MRCONSO (atoms/strings)…")
    n_atoms = load_mrconso(conn, args.rrf_dir)
    print(f"  {n_atoms:,} atom rows loaded")

    print("Loading MRSTY (semantic types)…")
    n_sty = load_mrsty(conn, args.rrf_dir)
    print(f"  {n_sty:,} semtype rows loaded")

    print("Loading MRDEF (definitions)…")
    n_def = load_mrdef(conn, args.rrf_dir)
    print(f"  {n_def:,} definition rows loaded")

    print("Building indices…")
    _build_indices(conn)
    conn.close()

    print(f"\nDone → {args.output}")
    print("Next: python -m medjargone.pipeline.run --model <qwen_path> --input report.txt")


if __name__ == "__main__":
    main()
