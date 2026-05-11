#!/usr/bin/env python3
"""
preprocess_ctu13.py
───────────────────
Convert CTU-13 binetflow files into the feature schema expected by train.py.

CTU-13 is a 13-scenario botnet capture dataset from CTU University, Prague.
Each scenario is a different malware family (Neris, Rbot, Virut, Murlo, etc.),
which is exactly the cross-family diversity CIC-IDS-2017 lacks. Training on
CTU-13's 13 scenarios prevents single-family memorization.

Format
──────
.binetflow files are CSVs produced by Argus with 15 columns:

    StartTime, Dur, Proto, SrcAddr, Sport, Dir, DstAddr, Dport,
    State, sTos, dTos, TotPkts, TotBytes, SrcBytes, Label

Labels follow this convention:
    - "flow=Background-..."           → noisy, exclude
    - "flow=Normal-..."               → benign (label 0)
    - "flow=From-Botnet-..."          → c2     (label 1)
    - "flow=To-Botnet-..."            → c2     (label 1)
    - "flow=From-Normal-V*-...Botnet" → c2     (label 1, infected normal hosts)

Background flows are excluded because Stratosphere did not manually label
them — they're traffic of unknown origin, treating them as benign would
inject label noise into the training set.

Mapping limitations
───────────────────
Argus binetflows preserve fewer fields than CICFlowMeter CSVs:
    - No packet-length distribution (mean/std/min/max per packet)
    - No IAT statistics (only flow duration)
    - No per-direction byte counts (only fwd via SrcBytes; bwd is derived)

This means a CTU-13-trained model has even less to work with at the feature
level than a CIC-trained one. The reason to use CTU-13 anyway is the
multi-family diversity — the model can't memorize a single port pattern or
byte distribution because 13 different malware families exhibit 13 different
profiles.

For richer features, the proper path is to process the raw PCAPs in each
scenario through this project's full parser.py → flow_builder.py → features.py
pipeline. preprocess_ctu13.py is the fast-path equivalent of preprocess_cic.py
that operates on the pre-extracted binetflows.

Usage
─────
    # All 13 scenarios concatenated
    python scripts/preprocess_ctu13.py \\
        --input "data/raw/CTU-13-Dataset/*/*.binetflow" \\
        --output data/processed/ctu13_all.csv \\
        --benign-sample 200000 \\
        -v

    # Specific scenario only
    python scripts/preprocess_ctu13.py \\
        --input "data/raw/CTU-13-Dataset/1/*.binetflow" \\
        --output data/processed/ctu13_neris.csv \\
        -v

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
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow `python scripts/preprocess_ctu13.py` to find the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c2classifier.features import FEATURE_NAMES

logger = logging.getLogger("preprocess_ctu13")

# ─────────────────────────────────────────────────────────────────────────────
# CTU-13 binetflow schema
# ─────────────────────────────────────────────────────────────────────────────

BINETFLOW_COLUMNS: List[str] = [
    "StartTime", "Dur", "Proto", "SrcAddr", "Sport", "Dir",
    "DstAddr", "Dport", "State", "sTos", "dTos",
    "TotPkts", "TotBytes", "SrcBytes", "Label",
]

# Protocol-name → IANA number map for the Proto column
# (CTU-13 uses lowercase strings, not numbers, unlike CIC)
PROTO_NAMES = {
    "tcp":   6,
    "udp":   17,
    "icmp":  1,
    "igmp":  2,
    "ipv6":  41,
    "ipv6-icmp": 58,
    "rtp":   0,    # not a real protocol number; mapped to 0
    "arp":   0,
    "pim":   103,
    "esp":   50,
    "ah":    51,
    "gre":   47,
    "ospf":  89,
    "ipnip": 4,
    "unas":  0,
    "udt":   0,
    "rtcp":  0,
}


# ─────────────────────────────────────────────────────────────────────────────
# Label classification
# ─────────────────────────────────────────────────────────────────────────────

def _classify_label(raw_label: str) -> Optional[int]:
    """
    Map a CTU-13 label string to a binary class:
        0 (benign), 1 (c2), or None (excluded — background).

    The string format is "flow=<Direction>-<Category>-...". Examples:
        flow=From-Botnet-V42-1-TCP-CC73          → c2
        flow=Background-Established              → excluded
        flow=Normal-V42-Stribrek                 → benign
        flow=From-Normal-V42-Stribrek-To-Botnet  → c2 (infected normal)
    """
    if not isinstance(raw_label, str):
        return None

    s = raw_label.strip()

    # Background flows have unknown ground truth — exclude them entirely
    if "Background" in s:
        return None

    # Any flow involving Botnet is c2 (regardless of direction)
    if "Botnet" in s:
        return 1

    # Normal flows are benign
    if "Normal" in s:
        return 0

    # Unknown label format — log once and exclude
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess_ctu13.py",
        description="Convert CTU-13 binetflow files into the c2-classifier feature schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        type=str, required=True,
        help="Path or glob matching one or more .binetflow files.",
    )
    p.add_argument(
        "--output", "-o",
        type=Path, required=True,
        help="Output CSV path.",
    )
    p.add_argument(
        "--benign-sample",
        type=int, default=None,
        help="Randomly subsample the benign (Normal) class to this many rows. "
             "CTU-13 has ~2M Normal flows across scenarios; 200000 is a good "
             "starting point for balanced training.",
    )
    p.add_argument(
        "--include-background",
        action="store_true",
        help="Include 'Background' flows as benign. NOT recommended — these "
             "were not manually labeled by Stratosphere and contain unknown "
             "traffic that may include unattributed C2.",
    )
    p.add_argument(
        "--scenario",
        type=int, default=None,
        help="Tag rows with the scenario number (1-13). Useful when concatenating "
             "multiple scenarios and wanting to do scenario-aware splitting later.",
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
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_inputs(pattern: str) -> List[Path]:
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no files matched: {pattern}")
    return [Path(m) for m in matches]


_SCENARIO_RE = re.compile(r"/(\d{1,2})/[^/]+\.binetflow$")


def _scenario_from_path(path: Path) -> Optional[int]:
    """
    Extract the CTU-13 scenario number (1-13) from a binetflow path.

    The canonical CTU-13 layout is:
        CTU-13-Dataset/<scenario>/capture<date>.binetflow

    Returns None if the scenario can't be identified.
    """
    match = _SCENARIO_RE.search(str(path))
    return int(match.group(1)) if match else None


def _load_one(path: Path) -> pd.DataFrame:
    """
    Load a single .binetflow file.

    Argus binetflows are comma-separated, headered, and use string protocols.
    No leading-whitespace quirks like CIC, but column order is fixed.
    """
    logger.info("loading %s", path)
    df = pd.read_csv(path, low_memory=False, encoding_errors="replace")

    # Some binetflows have lowercase or differently-cased headers
    df.columns = [c.strip() for c in df.columns]

    # Verify schema
    missing = [c for c in BINETFLOW_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("file %s missing columns: %s", path.name, missing)

    return df


def _load_all(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = _load_one(p)
        scenario = _scenario_from_path(p)
        if scenario is not None:
            df["_scenario"] = scenario
        else:
            df["_scenario"] = 0
        frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("concatenated %d files → %d rows", len(frames), len(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce numeric columns and drop rows with non-finite required values.
    """
    n_before = len(df)

    numeric_cols = ["Dur", "TotPkts", "TotBytes", "SrcBytes"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sport / Dport are often hex (e.g. "0x303f") for non-standard protocols
    # — convert to numeric, leaving NaN for unparseable values
    for port_col in ["Sport", "Dport"]:
        if port_col in df.columns:
            df[port_col] = df[port_col].apply(_parse_port)

    # Drop rows with NaN in required numeric columns
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Dur", "TotPkts", "TotBytes", "SrcBytes"])

    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("dropped %d rows with non-finite values (%.2f%%)",
                    n_dropped, 100.0 * n_dropped / max(n_before, 1))
    return df


def _parse_port(val) -> float:
    """Parse a port value that may be decimal, hex ('0x303f'), or missing."""
    if pd.isna(val):
        return 0.0
    try:
        s = str(val).strip()
        if s.startswith("0x") or s.startswith("0X"):
            return float(int(s, 16))
        return float(s)
    except (ValueError, TypeError):
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Filtering by label class
# ─────────────────────────────────────────────────────────────────────────────

def _filter_labels(
    df: pd.DataFrame,
    include_background: bool,
) -> pd.DataFrame:
    """
    Apply label classification and drop rows we can't use.

    Returns df with an added 'label' integer column.
    """
    n_before = len(df)
    df = df.copy()
    df["label"] = df["Label"].apply(_classify_label)

    if include_background:
        # Treat background as benign (NOT recommended)
        bg_mask = df["Label"].astype(str).str.contains("Background", na=False)
        df.loc[bg_mask & df["label"].isna(), "label"] = 0

    # Drop rows that didn't classify
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(np.int64)

    n_dropped = n_before - len(df)
    logger.info("label filtering: kept %d / %d rows (%d excluded)",
                len(df), n_before, n_dropped)

    n_benign = int((df["label"] == 0).sum())
    n_c2     = int((df["label"] == 1).sum())
    logger.info("  benign=%d, c2=%d (imbalance %.1f:1)",
                n_benign, n_c2, n_benign / max(n_c2, 1))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Transform to FEATURE_NAMES schema
# ─────────────────────────────────────────────────────────────────────────────

def _transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CTU-13 binetflow rows onto the project's 34-feature schema.

    Many features have no CTU-13 counterpart and are zero-filled. The features
    that DO get real values from CTU-13:
        - flow_duration       (from Dur, already in seconds — no conversion)
        - total_packets       (from TotPkts)
        - total_bytes         (from TotBytes)
        - fwd_total_bytes     (from SrcBytes)
        - bwd_total_bytes     (TotBytes - SrcBytes)
        - fwd_bwd_byte_ratio  (derived)
        - direction_dominance (derived)
        - pkts_per_second     (derived)
        - bytes_per_second    (derived)
        - avg_payload_size    (TotBytes / TotPkts)
        - protocol            (from Proto, mapped to IANA number)
        - dst_port            (from Dport)
    """
    out = pd.DataFrame(index=df.index)

    # ── Direct fields ───────────────────────────────────────────────────────
    out["flow_duration"]    = df["Dur"].astype(np.float64)
    out["total_packets"]    = df["TotPkts"].astype(np.float64)
    out["total_bytes"]      = df["TotBytes"].astype(np.float64)
    out["fwd_total_bytes"]  = df["SrcBytes"].astype(np.float64)
    out["bwd_total_bytes"]  = (df["TotBytes"] - df["SrcBytes"]).clip(lower=0).astype(np.float64)

    # ── Packet count split (heuristic: 60/40 fwd/bwd in absence of detail) ──
    # CTU-13 binetflows don't track per-direction packet counts. We approximate
    # by splitting TotPkts proportionally to the byte split.
    safe_total_b = out["total_bytes"].replace(0, np.nan)
    fwd_byte_frac = (out["fwd_total_bytes"] / safe_total_b).fillna(0.5)
    out["fwd_packet_count"] = (df["TotPkts"] * fwd_byte_frac).round().astype(np.float64)
    out["bwd_packet_count"] = (df["TotPkts"] - out["fwd_packet_count"]).clip(lower=0)

    # ── Derived ratios ──────────────────────────────────────────────────────
    out["fwd_bwd_byte_ratio"]   = (out["fwd_total_bytes"] / safe_total_b).fillna(0.0)
    out["direction_dominance"]  = (
        np.maximum(out["fwd_total_bytes"], out["bwd_total_bytes"]) / safe_total_b
    ).fillna(0.0)

    # ── Rates ───────────────────────────────────────────────────────────────
    safe_dur = out["flow_duration"].replace(0, np.nan)
    out["pkts_per_second"]  = (out["total_packets"] / safe_dur).fillna(0.0)
    out["bytes_per_second"] = (out["total_bytes"]   / safe_dur).fillna(0.0)

    # ── Average payload size ────────────────────────────────────────────────
    safe_pkts = out["total_packets"].replace(0, np.nan)
    out["avg_payload_size"] = (out["total_bytes"] / safe_pkts).fillna(0.0)

    # ── Protocol → IANA number ─────────────────────────────────────────────
    proto_series = df["Proto"].astype(str).str.lower().str.strip()
    out["protocol"] = proto_series.map(PROTO_NAMES).fillna(0).astype(np.float64)

    # ── Destination port ────────────────────────────────────────────────────
    out["dst_port"] = df["Dport"].astype(np.float64) if "Dport" in df.columns else 0.0

    # ── Features with no CTU-13 counterpart — zero-fill ─────────────────────
    zero_features = [
        "pkt_len_mean", "pkt_len_std", "pkt_len_min", "pkt_len_max",
        "small_pkt_ratio", "fwd_payload_bytes",
        "iat_mean", "iat_std", "iat_min", "iat_max",
        "iat_skew", "iat_kurtosis", "beacon_score", "active_time",
        "payload_entropy", "header_entropy", "dns_query_entropy",
        "fwd_syn_flag", "fwd_fin_flag", "rst_flag",
    ]
    for name in zero_features:
        out[name] = 0.0

    # ── Sanity check ────────────────────────────────────────────────────────
    missing = [n for n in FEATURE_NAMES if n not in out.columns]
    if missing:
        raise RuntimeError(f"internal error: features not produced: {missing}")

    # Reorder to canonical schema
    out = out[FEATURE_NAMES].astype(np.float32)

    # ── Append label and metadata ──────────────────────────────────────────
    out["label"] = df["label"].astype(np.int64).values

    # Parse StartTime to epoch seconds for time-based splitting
    if "StartTime" in df.columns:
        ts = pd.to_datetime(df["StartTime"], errors="coerce")
        out["start_time"] = ts.astype("int64") / 1e9
        out["start_time"] = out["start_time"].fillna(0.0)
    else:
        out["start_time"] = 0.0

    if "_scenario" in df.columns:
        out["scenario"] = df["_scenario"].astype(np.int32).values

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Benign subsampling
# ─────────────────────────────────────────────────────────────────────────────

def _subsample_benign(
    df: pd.DataFrame,
    n_keep: int,
    random_state: int,
) -> pd.DataFrame:
    benign_mask = df["label"] == 0
    n_benign = int(benign_mask.sum())

    if n_keep >= n_benign:
        logger.info("--benign-sample %d ≥ %d benign rows; no subsampling",
                    n_keep, n_benign)
        return df

    benign = df[benign_mask].sample(n=n_keep, random_state=random_state)
    attacks = df[~benign_mask]
    out = pd.concat([benign, attacks], ignore_index=True)
    out = out.sort_values("start_time").reset_index(drop=True)
    logger.info("--benign-sample: kept %d benign / %d c2 (was %d benign)",
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
    if raw.empty:
        logger.error("no rows loaded")
        return 1

    # ── Clean ───────────────────────────────────────────────────────────────
    raw = _clean(raw)

    # ── Filter labels ───────────────────────────────────────────────────────
    raw = _filter_labels(raw, include_background=args.include_background)
    if raw.empty:
        logger.error("no rows remained after label filtering")
        return 1

    # ── Transform to FEATURE_NAMES schema ───────────────────────────────────
    out = _transform(raw)

    # ── Optional benign subsampling ─────────────────────────────────────────
    if args.benign_sample is not None:
        out = _subsample_benign(out, args.benign_sample, args.random_state)

    # ── Final summary ───────────────────────────────────────────────────────
    n_benign = int((out["label"] == 0).sum())
    n_c2     = int((out["label"] == 1).sum())
    if n_c2 == 0:
        logger.error("no c2 rows in output — model cannot be trained")
        return 1

    logger.info("final dataset: %d rows (benign=%d, c2=%d, imbalance=%.1f:1)",
                len(out), n_benign, n_c2,
                n_benign / max(n_c2, 1))

    # Scenario distribution (if available)
    if "scenario" in out.columns:
        scen_counts = out.groupby("scenario")["label"].agg(["count", "sum"])
        scen_counts.columns = ["total", "c2"]
        logger.info("scenarios represented:\n%s", scen_counts.to_string())

    # ── Write ───────────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("\n" + "─" * 60)
    print(f"preprocessed CSV written  : {args.output}")
    print(f"input files              : {len(paths)}")
    print(f"output rows              : {len(out)}")
    print(f"benign / c2              : {n_benign} / {n_c2}")
    print(f"imbalance                : {n_benign / max(n_c2, 1):.1f}:1")
    print(f"feature columns          : {len(FEATURE_NAMES)}")
    print(f"includes start_time      : {'yes' if (out['start_time'] > 0).any() else 'no'}")
    print("─" * 60)
    print("\nNext step:")
    print(f"  python scripts/train.py \\")
    print(f"    --csv {args.output} \\")
    print(f"    --model models/rf_v3.joblib \\")
    print(f"    --split time \\")
    print(f"    --dataset-name 'CTU-13 (multi-family)' \\")
    print(f"    -v")

    return 0


if __name__ == "__main__":
    sys.exit(main())