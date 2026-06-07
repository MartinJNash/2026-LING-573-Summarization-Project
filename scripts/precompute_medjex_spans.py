"""
Standalone MedJEx jargon-span precomputation pass.

Runs the released MedJEx BERT-CRF model on the MultiClinSum corpus and writes
a JSON of {doc_id: [{term, start, end}]} for use by glossary.py.

Requires a separate env (see scripts/setup_medjex_env_hyak.sh).

Download the checkpoint (~500 MB, done automatically on first run).

Usage:
    python scripts/precompute_medjex_spans.py --split test \
        --medcat-path /gscratch/scrubbed/<netid>/medcat/umls_sm_wstatus_2021_oct

    # If checkpoint flags show no UMLS features needed:
    python scripts/precompute_medjex_spans.py --split test --no-umls
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── Path setup: vendor the MedJEx repo modules ───────────────────────────────
MEDJEX_DIR = Path(__file__).parents[1] / "src" / "medjargone" / "MedJEx"
if str(MEDJEX_DIR) not in sys.path:
    sys.path.insert(0, str(MEDJEX_DIR))

# ── Imports from vendored MedJEx ──────────────────────────────────────────────
import random
import re
import urllib.request

import numpy as np
import torch

# Fix for 3-year-old code: disable CUDA env overrides before torch import
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

from nltk.tokenize import word_tokenize, sent_tokenize

from loader.loader import load_file, load_data, load_ner_file
from utils.sequence_labeler import SequenceLabeler
from utils.term_weighting import TermFrequency, MLM_weight
from utils.normalization import Normalizer
from models.models import (
    EarlyAndLateFusionBertMLPCRF,
    EarlyAndLateFusionrobertaMLPCRF,
)
from transformers import AutoTokenizer, AutoConfig

from torch.utils.data import Dataset, DataLoader


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME      = "dmis-lab/biobert-base-cased-v1.1"
CHECKPOINT_URL  = "https://huggingface.co/Mozzi/MedJEx/resolve/main/model.pth"
CHECKPOINT_PATH = MEDJEX_DIR / "results" / "MedJEx.pth"
TEMP_TSV        = MEDJEX_DIR / "temp" / "temp.tsv"
MAX_TOKEN_LEN   = 128


# ── Dataset (copied from notebook, unchanged) ─────────────────────────────────

class JargonTerm(Dataset):
    def __init__(self, data, tokenizer, labeler, max_len,
                 Binary_flag, TF_flag, MLM_flag, UMLS_labeler):
        self.data = data
        self.pad_id = tokenizer.pad_token_id
        self.outter_id = labeler.label2id["O"]
        self.MAX_TOKEN_LEN = max_len
        self.TF_flag = TF_flag
        self.MLM_flag = MLM_flag
        self.Binary_flag = Binary_flag
        self.additional_feature_flag = Binary_flag or TF_flag or MLM_flag
        flags = [TF_flag, MLM_flag]
        self.num_of_binary_features  = len(UMLS_labeler.label2id) if Binary_flag else 0
        self.num_of_weighted_features = sum(
            len(UMLS_labeler.label2id) for f in flags if f
        )
        self._UMLS_labeler = UMLS_labeler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        loaded = self.data[idx]
        n = len(loaded["token_ids"])
        if n < self.MAX_TOKEN_LEN:
            pad = self.MAX_TOKEN_LEN - n
            input_ids      = loaded["token_ids"] + [self.pad_id] * pad
            label_ids      = loaded["yids"]      + [self.outter_id] * pad
            attention_mask = [1] * n + [0] * pad
        else:
            input_ids      = loaded["token_ids"][:self.MAX_TOKEN_LEN]
            label_ids      = loaded["yids"][:self.MAX_TOKEN_LEN]
            attention_mask = [1] * self.MAX_TOKEN_LEN
        token_type_ids = [0] * self.MAX_TOKEN_LEN

        item = {
            "input_ids":      torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "labels":         torch.tensor(label_ids),
            "sentid":         loaded["sentid"],
        }

        if self.additional_feature_flag:
            item["bin_weights"] = self._binary_feature(loaded)
            if self.TF_flag:
                item["TF_weights"]  = self._weighted_feature(loaded, "term_frequency")
            if self.MLM_flag:
                item["MLM_weights"] = self._weighted_feature(loaded, "MLM_weight")
        else:
            item["additional_features"] = {}
        return item

    def _binary_feature(self, loaded):
        dim = len(self._UMLS_labeler.label2id)
        rep = np.zeros((self.MAX_TOKEN_LEN, dim))
        for i in range(self.MAX_TOKEN_LEN):
            rep[i][self._UMLS_labeler.label2id["O"]] = 1.0
        for concept in loaded["UMLS_concepts"]:
            for tok_id in range(concept["start_token"], concept["end_token"]):
                if tok_id < self.MAX_TOKEN_LEN:
                    rep[tok_id] = 0.0
                    rep[tok_id][self._UMLS_labeler.label2id["O"]] = 0.0
        return torch.tensor(rep, dtype=torch.float32)

    def _weighted_feature(self, loaded, feature):
        dim = len(self._UMLS_labeler.label2id)
        rep = np.zeros((self.MAX_TOKEN_LEN, dim))
        for concept in loaded["UMLS_concepts"]:
            w = concept.get(feature, 0.0)
            for tok_id in range(concept["start_token"], concept["end_token"]):
                if tok_id < self.MAX_TOKEN_LEN:
                    rep[tok_id][self._UMLS_labeler.label2id["O"]] = 0.0
        return torch.tensor(rep, dtype=torch.float32)


# ── BIOES decoder — extended to return token spans ───────────────────────────

def bioes_decode(tokens, labels, tokenizer):
    """Returns list of {text, start_token, end_token}."""
    entities = []
    inner = False
    for i, (tok, lbl) in enumerate(zip(tokens, labels)):
        ltype = lbl[0] if lbl else "O"
        etype = lbl[2:] if len(lbl) > 2 else ""
        if not inner:
            if ltype == "B":
                inner = (i, etype)
            elif ltype == "S":
                text = tokenizer.convert_tokens_to_string(tokens[i:i+1])
                entities.append({"text": text, "start_token": i, "end_token": i+1})
        else:
            if ltype in ("B", "S") or (ltype not in ("I", "E") and ltype != inner[1]):
                inner = False
            elif ltype == "E" and etype == inner[1]:
                text = tokenizer.convert_tokens_to_string(tokens[inner[0]:i+1])
                entities.append({"text": text, "start_token": inner[0], "end_token": i+1})
                inner = False
    return entities


# ── Char-offset recovery ──────────────────────────────────────────────────────

def find_all_offsets(term: str, source: str) -> list[tuple[int, int]]:
    """All (start, end) char positions of term in source (case-insensitive)."""
    results = []
    lo = source.lower()
    lt = term.lower().strip()
    if not lt:
        return results
    start = 0
    while True:
        idx = lo.find(lt, start)
        if idx == -1:
            break
        results.append((idx, idx + len(lt)))
        start = idx + 1
    return results


def spans_with_offsets(jargon_strings: set[str], source: str) -> list[dict]:
    """Convert MedJEx string set → [{term, start, end}], deduplicated."""
    seen: set[tuple[int, int]] = set()
    out = []
    for term in sorted(jargon_strings):
        for s, e in find_all_offsets(term, source):
            if (s, e) not in seen:
                seen.add((s, e))
                out.append({"term": source[s:e], "start": s, "end": e})
    return sorted(out, key=lambda x: x["start"])


# ── MedJEx predictor ──────────────────────────────────────────────────────────

class MedJExPredictor:
    def __init__(self, checkpoint: Path, umls_matcher, device: torch.device,
                 ignore_mlm: bool = False):
        self.device       = device
        self.ignore_mlm   = ignore_mlm
        self.umls_matcher = umls_matcher

        print(f"Loading tokenizer: {MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.BIOES_labeler = SequenceLabeler(labeling_scheme="BIOES",
                                              longest_labeling_flag=True)
        self.UMLS_labeler  = SequenceLabeler(labeling_scheme="BIOES",
                                              longest_labeling_flag=True)

        print(f"Loading checkpoint: {checkpoint}")
        # Fix for old torch: add map_location + weights_only=False
        ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
        self.ckpt = ckpt

        Binary_flag      = ckpt["Binary_flag"]
        TF_flag          = ckpt["TF_flag"]
        MLM_flag         = ckpt["MLM_flag"]
        additional_feat  = ckpt["additional_feature"]
        self.Binary_flag = Binary_flag
        self.TF_flag     = TF_flag
        self.MLM_flag    = MLM_flag

        num_add = None
        if additional_feat:
            dummy_ds = self._make_dummy_dataset(Binary_flag, TF_flag, MLM_flag)
            num_add = {
                "num_of_binary_features":   dummy_ds.num_of_binary_features,
                "num_of_weighted_features": dummy_ds.num_of_weighted_features,
            }
            if not TF_flag and not MLM_flag:
                num_add["num_of_weighted_features"] = 0

        cfg = AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=self.BIOES_labeler.num_of_label,
        )
        if "roberta" in MODEL_NAME.lower():
            model_cls = EarlyAndLateFusionrobertaMLPCRF
        else:
            model_cls = EarlyAndLateFusionBertMLPCRF

        self.model = model_cls.from_pretrained(
            MODEL_NAME, config=cfg,
            num_of_additional_features=num_add,
        )
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.model.to(device)
        self.model.eval()

        self.TF_weighter  = TermFrequency()
        self.MLM_weighter = MLM_weight(MODEL_NAME)

    def _make_dummy_dataset(self, B, T, M):
        """One-item dataset just to read feature dimensions."""
        dummy = [{"token_ids": [0], "yids": [0], "sentid": "x", "UMLS_concepts": []}]
        return JargonTerm(dummy, self.tokenizer, self.BIOES_labeler,
                          MAX_TOKEN_LEN, B, T, M, self.UMLS_labeler)

    def _sentencify(self, text: str) -> list[str]:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]

    def _load_sents(self, sents: list[str]) -> dict:
        TEMP_TSV.parent.mkdir(parents=True, exist_ok=True)
        with open(TEMP_TSV, "w") as f:
            for sent in sents:
                for word in word_tokenize(sent):
                    f.write(f"{word}\tO\n")
                f.write("\n")
        return load_ner_file(str(TEMP_TSV))

    def predict(self, source_text: str, batch_size: int = 4) -> set[str]:
        sents = self._sentencify(source_text)
        note_dict = self._load_sents(sents)
        proc = load_data(note_dict, self.tokenizer, self.BIOES_labeler,
                         UMLS_matcher=self.umls_matcher,
                         UMLS_labeler=self.UMLS_labeler)

        split = [proc[k] for k in proc if len(proc[k]["token_ids"]) > 0]

        split = self.TF_weighter.get_weights(split, target_entities="entities")
        split = self.TF_weighter.get_weights(split, target_entities="UMLS_concepts")
        split = self.MLM_weighter.get_weights(
            split, target_entities="entities",
            ignore=self.ignore_mlm,
        )
        split = self.MLM_weighter.get_weights(
            split, target_entities="UMLS_concepts",
            ignore=self.ignore_mlm,
        )

        normalizer = Normalizer(split, normalization_type="min_max")
        split = self._normalize(split, normalizer)

        ds = JargonTerm(split, self.tokenizer, self.BIOES_labeler,
                        MAX_TOKEN_LEN,
                        self.Binary_flag, self.TF_flag, self.MLM_flag,
                        self.UMLS_labeler)
        loader = DataLoader(ds, batch_size=batch_size)

        jargon_strings: set[str] = set()

        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                labels         = batch["labels"].to(self.device)

                add_feats: dict = {}
                if self.Binary_flag or self.TF_flag or self.MLM_flag:
                    add_feats["binary_features"] = batch["bin_weights"].to(self.device)
                    weighted = torch.tensor([])
                    if self.TF_flag:
                        weighted = torch.cat((weighted, batch["TF_weights"]), dim=-1)
                    if self.MLM_flag:
                        weighted = torch.cat((weighted, batch["MLM_weights"]), dim=-1)
                    add_feats["weighted_features"] = weighted.to(self.device)

                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    token_type_ids=token_type_ids,
                    additional_features=add_feats,
                )
                _, preds = outputs[0], outputs[1]

                np_ids = input_ids.cpu().numpy()
                for b_idx, pred in enumerate(preds):
                    tok_labels = [self.BIOES_labeler.id2label[p] for p in pred]
                    tok_strings = [
                        self.tokenizer.convert_ids_to_tokens([np_ids[b_idx][i]])[0]
                        for i in range(len(tok_labels))
                    ]
                    for ent in bioes_decode(tok_strings, tok_labels, self.tokenizer):
                        t = ent["text"].strip()
                        if t:
                            jargon_strings.add(t)

        return jargon_strings

    def _normalize(self, split, normalizer):
        for target in ("UMLS_concepts", "entities"):
            for item in split:
                for c in item[target]:
                    c["term_frequency"] = normalizer.normalizer[
                        "term_frequency"
                    ].get_normalized_results(c["term_frequency"])
                    c["MLM_weight"] = normalizer.normalizer[
                        "MLM_weight"
                    ].get_normalized_results(c["MLM_weight"])
        return split


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_corpus(split: str, data_root: Path | None = None) -> list[dict]:
    """
    Load from local data/multiclinsum_{split}_en/fulltext/.
    Files are named multiclinsum_{split}_{file_id}_en.txt with file_id starting at 1.
    Pipeline IDs are 0-indexed: pipeline_id = file_id - 1.
    """
    if data_root is None:
        data_root = Path(__file__).parents[1] / "data"
    fulltext_dir = data_root / f"multiclinsum_{split}_en" / "fulltext"
    if not fulltext_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {fulltext_dir}")

    docs = []
    for fpath in sorted(fulltext_dir.glob(f"multiclinsum_{split}_*_en.txt"),
                        key=lambda p: int(p.stem.split("_")[-2])):
        file_id = int(fpath.stem.split("_")[-2])
        pipeline_id = file_id - 1
        docs.append({"id": str(pipeline_id), "input": fpath.read_text(encoding="utf-8")})
    return docs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute MedJEx jargon spans")
    root = Path(__file__).parents[1]
    parser.add_argument("--split",        default="test")
    parser.add_argument("--output",       default=str(root / "data/medjargone/medjex_spans_test.json"))
    parser.add_argument("--checkpoint",   default=str(root / "src/medjargone_d4/MedJEx/results/MedJEx.pth"))
    parser.add_argument("--data-root",    default=str(root / "data"),
                        help="Root of data/ directory")
    parser.add_argument("--medcat-path",  default=None,
                        help="Path to MedCAT model pack dir")
    parser.add_argument("--no-umls",      action="store_true",
                        help="Run without UMLS matcher (valid only if checkpoint "
                             "Binary_flag/TF_flag/MLM_flag are all False)")
    parser.add_argument("--batch-size",   type=int, default=4)
    parser.add_argument("--ignore-mlm",   action="store_true",
                        help="Zero out MLM features (faster, may hurt detection)")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--start-idx",    type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Download checkpoint if missing
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Downloading MedJEx checkpoint → {ckpt_path}")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CHECKPOINT_URL, str(ckpt_path))

    # UMLS matcher
    if args.medcat_path:
        from utils.MedCAT import MedCAT_wrapper
        umls_matcher = MedCAT_wrapper(args.medcat_path)
        print(f"Using MedCAT: {args.medcat_path}")
    elif args.no_umls:
        umls_matcher = None
        print("No UMLS matcher (--no-umls); will validate against checkpoint flags")
    else:
        parser.error("Provide --medcat-path or --no-umls")

    # Peek at checkpoint flags before building the full model
    if args.no_umls:
        _ckpt_peek = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        needs_umls = (_ckpt_peek.get("Binary_flag") or
                      _ckpt_peek.get("TF_flag") or
                      _ckpt_peek.get("MLM_flag"))
        if needs_umls:
            flags = {k: _ckpt_peek.get(k) for k in ("Binary_flag", "TF_flag", "MLM_flag")}
            raise SystemExit(f"Checkpoint requires UMLS features {flags}; "
                             "provide --medcat-path instead of --no-umls")
        del _ckpt_peek

    predictor = MedJExPredictor(
        checkpoint=ckpt_path,
        umls_matcher=umls_matcher,
        device=device,
        ignore_mlm=args.ignore_mlm,
    )

    print(f"Loading corpus: {args.split}")
    corpus = load_corpus(args.split, data_root=Path(args.data_root))
    end = (args.start_idx + args.num_examples
           if args.num_examples else len(corpus))
    corpus = corpus[args.start_idx:end]
    print(f"Processing {len(corpus)} documents (idx {args.start_idx}–{end-1})")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing output to allow resuming
    results: dict[str, list] = {}
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} docs already done")

    for i, doc in enumerate(corpus):
        doc_id = doc["id"]
        if doc_id in results:
            continue
        try:
            jargon_strs = predictor.predict(doc["input"], batch_size=args.batch_size)
            results[doc_id] = spans_with_offsets(jargon_strs, doc["input"])
        except Exception as exc:
            print(f"\n  [warn] doc {doc_id} failed: {exc}")
            results[doc_id] = []

        if (i + 1) % 10 == 0 or i == 0:
            with open(out_path, "w") as f:
                json.dump(results, f)
            print(f"  {i+1}/{len(corpus)} — saved checkpoint", flush=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {len(results)} docs → {out_path}")


if __name__ == "__main__":
    main()
