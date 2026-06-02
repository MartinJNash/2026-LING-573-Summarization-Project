"""
Build a MedCAT v2 model pack from UMLS MRCONSO.RRF.

Produces a model pack at /gscratch/scrubbed/pgarg2/medcat/umls_2026AA_<hash>/
which can be passed directly to --medcat-path in precompute_medjex_spans.py.

Usage:
    python scripts/build_medcat_cdb.py
    python scripts/build_medcat_cdb.py \
        --mrconso /gscratch/scrubbed/pgarg2/umls/2026AA/META/MRCONSO.RRF \
        --output  /gscratch/scrubbed/pgarg2/medcat
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from medcat.cdb import CDB
from medcat.config import Config
from medcat.model_creation.cdb_maker import CDBMaker
from medcat.vocab import Vocab
from medcat.cat import CAT


# MRCONSO.RRF has 18 pipe-separated fields then a trailing |
# Columns we actually need:
#   0  CUI   – concept unique identifier
#   1  LAT   – language (keep ENG only)
#   6  ISPREF– Y = preferred name for this concept
#  14  STR   – the actual string
#  16  SUPPRESS – N = not suppressed
_MRCONSO_COLS = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI",
    "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL",
    "SUPPRESS", "CVF",
]
_KEEP_COLS = {"CUI", "LAT", "ISPREF", "STR", "SUPPRESS"}
_CHUNK = 500_000


def load_mrconso(path: Path) -> pd.DataFrame:
    print(f"Reading {path} …")
    kept: list[pd.DataFrame] = []
    total = 0
    for chunk in pd.read_csv(
        path,
        sep="|",
        header=None,
        names=_MRCONSO_COLS,
        dtype=str,
        chunksize=_CHUNK,
        index_col=False,      # handles trailing | at end of each line
        usecols=list(range(18)),
    ):
        total += len(chunk)
        filt = chunk[
            (chunk["LAT"] == "ENG") &
            (chunk["SUPPRESS"] == "N")
        ][["CUI", "STR", "ISPREF"]]
        kept.append(filt)
        if (total // _CHUNK) % 5 == 0:
            print(f"  {total:,} rows read, {sum(len(k) for k in kept):,} kept", flush=True)

    df = pd.concat(kept, ignore_index=True)
    print(f"Done: {total:,} total rows → {len(df):,} English non-suppressed")
    return df


def build_cdb(df: pd.DataFrame) -> tuple[CDB, Config]:
    cdb_input = pd.DataFrame({
        "cui":         df["CUI"],
        "name":        df["STR"],
        # ISPREF=Y → preferred name (P), everything else → acceptable (A)
        "name_status": df["ISPREF"].map({"Y": "P"}).fillna("A"),
    })
    print(f"Building CDB from {len(cdb_input):,} name entries …", flush=True)
    cnf = Config()
    maker = CDBMaker(cnf)
    maker.prepare_csvs([cdb_input], full_build=True)
    cdb = maker.cdb
    print(f"CDB built: {len(cdb.cui2info):,} concepts")
    return cdb, cnf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mrconso",
        default="/gscratch/scrubbed/pgarg2/umls/2026AA/META/MRCONSO.RRF",
    )
    parser.add_argument(
        "--output",
        default="/gscratch/scrubbed/pgarg2/medcat",
    )
    args = parser.parse_args()

    mrconso_path = Path(args.mrconso)
    output_dir   = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df  = load_mrconso(mrconso_path)
    cdb, cnf = build_cdb(df)

    print("Creating model pack …", flush=True)
    vocab = Vocab()
    cat   = CAT(cdb=cdb, config=cnf, vocab=vocab)
    pack  = cat.save_model_pack(
        str(output_dir),
        pack_name="umls_2026AA",
        add_hash_to_pack_name=True,
    )
    print(f"\nModel pack saved → {pack}")
    print("Pass this path to --medcat-path in precompute_medjex_spans.py")


if __name__ == "__main__":
    main()
