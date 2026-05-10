#!/usr/bin/env python3
"""
preprocess_cic.py
─────────────────
Convert CIC-IDS-2017 (CICFlowMeter) CSVs into the feature schema expected
by train.py.

CIC-IDS-2017 ships ~80 features per flow extracted by CICFlowMeter. Our
classifier defines its own 34-feature schema (see features.FEATURE_NAMES).
This script:

    1. Loads one or more CIC CSV files (the dataset ships split by day)
    2. Normalises column names (CIC has known leading-whitespace bugs)
    3. Maps CIC columns to our FEATURE_NAMES, deriving missing features
       and zeroing features that have no CIC counterpart (entropy, skew,
       kurtosis — these are 0 here and only become useful when training
       on raw PCAPs through your own pipeline)
    4. Cleans known data-quality issues (inf in Flow Bytes/s on zero-duration
       flows, NaN scattered through several columns)
    5. Maps labels: 'BENIGN' → 0, everything else → 1 (binary classification)
    6. Optionally filters to a single attack family (e.g. --only-botnet) for
       focused C2 detection training
    7. Optionally subsamples the (heavily imbalanced) benign class for speed
    8. Writes a single CSV with FEATURE_NAMES columns + 'label' + 'start_time'

Important caveat
────────────────
CIC-IDS-2017 has documented data-quality issues — duplicate flows, label
errors, incomplete attack labelling. See:
    "Errors in the CICIDS2017 Dataset and the Significant Differences in
     Detection Performances It Makes" (2022)

The preprocessing here cleans the most egregious problems but does NOT
attempt to relabel anything. A model trained on this data will inherit the
dataset's label noise. For high-stakes evaluation, supplement with CTU-13
or your own labelled lab captures.

Usage
─────
    # Single file
    python scripts/preprocess_cic.py \\
        --input data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv \\
        --output data/processed/cic_friday_morning.csv

    # Glob over the whole MachineLearningCSV dataset
    python scripts/preprocess_cic.py \\
        --input "data/raw/MachineLearningCSV/*.csv" \\
        --output data/processed/cic_ids_2017.csv

    # Botnet-only subset (most C2-relevant attack family in CIC)
    python scripts/preprocess_cic.py \\
        --input "data/raw/MachineLearningCSV/*.csv" \\
        --output data/processed/cic_botnet.csv \\
        --only-botnet

    # Subsample benign class to speed up training iteration
    python scripts/preprocess_cic.py \\
        --input "data/raw/MachineLearningCSV/*.csv" \\
        --output data/processed/cic_balanced.csv \\
        --benign-sample 100000

Exit codes
──────────
    0 — preprocessed CSV written
    1 — runtime error
    2 — invalid arguments
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Allow `python scripts/preprocess_cic.py` to find the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c2classifier.features import FEATURE_NAMES

logger = logging.getLogger("preprocess_cic")

# ─────────────────────────────────────────────────────────────────────────────
# CIC column mapping
# ─────────────────────────────────────────────────────────────────────────────
#
# CIC-IDS-2017 timestamps and IATs are in MICROSECONDS — divide by 1e6 to
# get our seconds-based schema.

# Direct 1:1 mappings (after a /1e6 conversion where noted in CONVERT_TO_SECONDS)
CIC_DIRECT_MAP: Dict[str, str] = {
    # our_name              : cic_column
    "flow_duration"         : "Flow Duration",            # μs → s
    "fwd_packet_count"      : "Total Fwd Packets",
    "bwd_packet_count"      : "Total Backward Packets",
    "fwd_total_bytes"       : "Total Length of Fwd Packets",
    "bwd_total_bytes"       : "Total Length of Bwd Packets",
    "pkt_len_mean"          : "Packet Length Mean",
    "pkt_len_std"           : "Packet Length Std",
    "pkt_len_min"           : "Min Packet Length",
    "pkt_len_max"           : "Max Packet Length",
    "iat_mean"              : "Flow IAT Mean",            # μs → s
    "iat_std"               : "Flow IAT Std",             # μs → s
    "iat_min"               : "Flow IAT Min",             # μs → s
    "iat_max"               : "Flow IAT Max",             # μs → s
    "pkts_per_second"       : "Flow Packets/s",
    "bytes_per_second"      : "Flow Bytes/s",
    "avg_payload_size"      : "Average Packet Size",
    "dst_port"              : "Destination Port",
    "active_time"           : "Active Mean",                  # μs → s
}

# Columns that need μs → s conversion
CONVERT_TO_SECONDS = {
    "flow_duration",
    "iat_mean", "iat_std", "iat_min", "iat_max",
    "active_time",
}

# Features with no CIC counterpart — set to 0.0
# (entropy / skewness / kurtosis require the raw payload + IAT sequence,
#  which CIC's pre-extracted CSVs throw away)
FEATURES_ZERO: List[str] = [
    "iat_skew", "iat_kurtosis",
    "payload_entropy", "header_entropy", "dns_query_entropy",
    "small_pkt_ratio",
    "active_time",
    "fwd_payload_bytes",
    "protocol",  # CIC MachineLearningCVE drops this column
]

# Label normalisation: BENIGN → 0, everything else → 1
def _normalise_label(raw: str) -> int:
    """
    Binary label mapping: BENIGN → 0, any attack → 1.

    Strips whitespace and unicode dashes which CIC sometimes uses
    (e.g. "Web Attack – Brute Force" with an en-dash).
    """
    s = str(raw).strip().upper()
    if s in ("BENIGN", "NORMAL", "BACKGROUND", "0"):
        return 0
    return 1


# Botnet label values seen in CIC-IDS-2017 (case-insensitive)
BOTNET_LABELS = {"BOT", "BOTNET", "BOTNET ARES"}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess_cic.py",
        description="Convert CIC-IDS-2017 CSVs into the c2-classifier feature schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        type=str, required=True,
        help="Path or glob pattern matching one or more CIC CSV files.",
    )
    p.add_argument(
        "--output", "-o",
        type=Path, required=True,
        help="Output CSV path.",
    )
    p.add_argument(
        "--only-botnet",
        action="store_true",
        help="Keep only Botnet attack flows (drop other attack families). "
             "BENIGN flows are still kept. Useful for focused C2-style "
             "training because CIC's botnet traffic is the closest analogue "
             "to real C2 in the dataset.",
    )
    p.add_argument(
        "--benign-sample",
        type=int, default=None,
        help="Randomly subsample the BENIGN class to this many rows. "
             "Helps with class imbalance and shrinks the training set for "
             "faster iteration. Attack rows are never subsampled.",
    )
    p.add_argument(
        "--random-state",
        type=int, default=42,
        help="Random seed used for benign subsampling. Default: 42.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="count", default=0,
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Loading and cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_inputs(pattern: str) -> List[Path]:
    """
    Resolve a glob or single path into a list of CSV files.

    CIC's MachineLearningCSV.zip unzips to a directory named
    'MachineLearningCVE' (their internal typo). We try the literal pattern
    first, then attempt the CSV↔CVE swap as a fallback before failing.
    """
    matches = sorted(glob.glob(pattern))

    if not matches and "MachineLearningCSV" in pattern:
        alt = pattern.replace("MachineLearningCSV", "MachineLearningCVE")
        matches = sorted(glob.glob(alt))
        if matches:
            logger.info("auto-corrected path: %s → %s", pattern, alt)

    if not matches and "MachineLearningCVE" in pattern:
        alt = pattern.replace("MachineLearningCVE", "MachineLearningCSV")
        matches = sorted(glob.glob(alt))
        if matches:
            logger.info("auto-corrected path: %s → %s", pattern, alt)

    if not matches:
        raise FileNotFoundError(f"no files matched: {pattern}")
    return [Path(m) for m in matches]


def _load_one(path: Path) -> pd.DataFrame:
    """Load a single CIC CSV with column-name normalisation."""
    logger.info("loading %s", path)

    # CIC files are ~150-300 MB each; let pandas infer dtypes but coerce
    # numerics aggressively after.
    # 'low_memory=False' avoids dtype-inference column splits.
    df = pd.read_csv(path, low_memory=False, encoding_errors="replace")

    # CIC has a known bug: every column name is prefixed with a leading
    # space. Strip whitespace + collapse internal multi-spaces.
    df.columns = [" ".join(c.split()).strip() for c in df.columns]

    return df


def _load_all(paths: List[Path]) -> pd.DataFrame:
    """Concatenate multiple CIC CSVs after column normalisation."""
    frames = [_load_one(p) for p in paths]
    df = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("concatenated %d files → %d rows", len(frames), len(df))
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with non-finite numeric values.

    CIC-IDS-2017 has documented inf values in 'Flow Bytes/s' and
    'Flow Packets/s' on flows with zero duration, and scattered NaNs
    in several columns. Rather than imputing, we drop — there are
    millions of clean rows and fabricated values would inject
    misleading signal.
    """
    n_before = len(df)

    # Convert all known numeric columns to numeric dtype, coercing errors to NaN
    numeric_cols = list(CIC_DIRECT_MAP.values())
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace ±inf with NaN, then drop any row with NaN in our required cols
    required = [c for c in numeric_cols if c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("dropped %d rows with non-finite values (%.2f%%)",
                    n_dropped, 100.0 * n_dropped / max(n_before, 1))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature transformation
# ─────────────────────────────────────────────────────────────────────────────

def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map a cleaned CIC dataframe onto our FEATURE_NAMES schema.

    Returns a new DataFrame with exactly:
        FEATURE_NAMES + ["label", "start_time"]
    """
    out = pd.DataFrame(index=df.index)

    # ── 1. Direct mappings ──────────────────────────────────────────────────
    for our_name, cic_col in CIC_DIRECT_MAP.items():
        if cic_col not in df.columns:
            logger.warning("CIC column missing: %r — filling %r with 0",
                           cic_col, our_name)
            out[our_name] = 0.0
            continue
        series = df[cic_col].astype(np.float64)
        if our_name in CONVERT_TO_SECONDS:
            series = series / 1_000_000.0  # μs → s
        out[our_name] = series

    # ── 2. Derived features ─────────────────────────────────────────────────
    fwd_b = out["fwd_total_bytes"]
    bwd_b = out["bwd_total_bytes"]
    total_b = fwd_b + bwd_b
    total_p = out["fwd_packet_count"] + out["bwd_packet_count"]

    out["total_packets"] = total_p
    out["total_bytes"]   = total_b

    # Avoid /0 — match _safe_div semantics from features.py
    safe_total_b = total_b.replace(0, np.nan)
    out["fwd_bwd_byte_ratio"]   = (fwd_b / safe_total_b).fillna(0.0)
    out["direction_dominance"]  = (
        np.maximum(fwd_b, bwd_b) / safe_total_b
    ).fillna(0.0)

    # Beacon score: CV of IAT (std/mean). Low CV → periodic beacon.
    safe_iat_mean = out["iat_mean"].replace(0, np.nan)
    out["beacon_score"] = (out["iat_std"] / safe_iat_mean).fillna(0.0)

    # ── 3. Flag features (CIC has SYN/FIN/RST counts; we want booleans) ─────
    flag_columns = {
        "fwd_syn_flag": "SYN Flag Count",
        "fwd_fin_flag": "FIN Flag Count",
        "rst_flag":     "RST Flag Count",
    }
    for our_name, cic_col in flag_columns.items():
        if cic_col in df.columns:
            counts = pd.to_numeric(df[cic_col], errors="coerce").fillna(0)
            out[our_name] = (counts > 0).astype(np.float32)
        else:
            out[our_name] = 0.0

    # ── 4. Zero-fill features with no CIC counterpart ───────────────────────
    for name in FEATURES_ZERO:
        out[name] = 0.0

    # ── 5. Sanity: ensure every FEATURE_NAMES column is present ─────────────
    missing = [n for n in FEATURE_NAMES if n not in out.columns]
    if missing:
        raise RuntimeError(
            f"internal error: {len(missing)} features not produced: {missing}"
        )

    # Reorder to canonical FEATURE_NAMES ordering
    out = out[FEATURE_NAMES].astype(np.float32)

    # ── 6. Labels ───────────────────────────────────────────────────────────
    if "Label" not in df.columns:
        raise ValueError("CIC CSV missing 'Label' column")
    out["label"] = df["Label"].apply(_normalise_label).astype(np.int64)

    # ── 7. Start time for chronological train/test splitting ────────────────
    if "Timestamp" in df.columns:
        # CIC timestamps look like "6/7/2017 9:15" — parse with dayfirst.
        ts = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=False)
        out["start_time"] = ts.astype("int64") / 1e9   # ns → s since epoch
        # Replace NaT (parse failures) with 0
        out["start_time"] = out["start_time"].fillna(0.0)
    else:
        logger.warning("CIC CSV missing 'Timestamp' column; "
                       "time-based splitting will not be available")
        out["start_time"] = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Filtering helpers
# ─────────────────────────────────────────────────────────────────────────────

def _filter_botnet_only(df: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Keep BENIGN flows + only Botnet attack flows."""
    raw_labels = original["Label"].astype(str).str.strip().str.upper()
    is_benign  = raw_labels.isin({"BENIGN", "NORMAL", "BACKGROUND"})
    is_botnet  = raw_labels.isin(BOTNET_LABELS)
    keep = is_benign | is_botnet
    n_before = len(df)
    df = df[keep.values].reset_index(drop=True)
    logger.info("--only-botnet: kept %d / %d rows "
                "(benign=%d, botnet=%d)",
                len(df), n_before,
                int(is_benign.sum()), int(is_botnet.sum()))
    return df


def _subsample_benign(
    df: pd.DataFrame,
    n_keep: int,
    random_state: int,
) -> pd.DataFrame:
    """Randomly downsample the benign class to *n_keep* rows."""
    benign_mask = df["label"] == 0
    n_benign    = int(benign_mask.sum())

    if n_keep >= n_benign:
        logger.info("--benign-sample %d ≥ %d benign rows; no subsampling",
                    n_keep, n_benign)
        return df

    benign = df[benign_mask].sample(
        n=n_keep, random_state=random_state,
    )
    attacks = df[~benign_mask]
    out = pd.concat([benign, attacks], ignore_index=True)
    out = out.sort_values("start_time").reset_index(drop=True)
    logger.info("--benign-sample: kept %d benign / %d attacks (was %d benign)",
                len(benign), len(attacks), n_benign)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Resolve and load ────────────────────────────────────────────────────
    try:
        paths = _resolve_inputs(args.input)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("found %d input file(s)", len(paths))

    raw = _load_all(paths)

    # ── Filter to botnet-only if requested (operates on raw labels) ─────────
    if args.only_botnet:
        raw = _filter_botnet_only(raw, raw)

    # ── Clean and transform ─────────────────────────────────────────────────
    raw = _clean(raw)
    if raw.empty:
        logger.error("no rows survived cleaning")
        return 1

    out = _transform(raw)

    # ── Optional benign subsampling ─────────────────────────────────────────
    if args.benign_sample is not None:
        out = _subsample_benign(out, args.benign_sample, args.random_state)

    # ── Class balance summary ───────────────────────────────────────────────
    n_benign = int((out["label"] == 0).sum())
    n_c2     = int((out["label"] == 1).sum())
    logger.info("final dataset: %d rows (benign=%d, c2=%d, imbalance=%.1f:1)",
                len(out), n_benign, n_c2,
                n_benign / max(n_c2, 1))

    if n_c2 == 0:
        logger.error("no attack rows in output — model cannot be trained")
        return 1

    # ── Write ───────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("\n" + "─" * 60)
    print(f"preprocessed CSV written  : {args.output}")
    print(f"input files              : {len(paths)}")
    print(f"output rows              : {len(out)}")
    print(f"benign / c2              : {n_benign} / {n_c2}")
    print(f"feature columns          : {len(FEATURE_NAMES)}")
    print(f"includes start_time      : {'yes' if (out['start_time'] > 0).any() else 'no'}")
    print("─" * 60)
    print("\nNext step:")
    print(f"  python scripts/train.py \\")
    print(f"    --csv {args.output} \\")
    print(f"    --model models/rf_v1.joblib \\")
    print(f"    --split time \\")
    print(f"    --dataset-name 'CIC-IDS-2017' \\")
    print(f"    -v")

    return 0


if __name__ == "__main__":
    sys.exit(main())