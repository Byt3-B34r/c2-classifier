#!/usr/bin/env python3
"""
train.py
────────
Offline training pipeline for the C2 classifier.

Loads a labeled flow dataset (CSV or PCAP+labels), trains a tree-based
classifier, evaluates on a held-out split, and serialises the model bundle
to disk for use by classify.py.

Two input modes
───────────────
1. CSV mode (recommended for first training run)
   The CSV must contain at minimum:
     - a column for each name in features.FEATURE_NAMES
     - a `label` column with values in {benign, c2, 0, 1}
   Use this with pre-extracted CIC-IDS-2017 / CTU-13 flow CSVs after
   running them through scripts/preprocess_cic.py (TODO).

2. PCAP+labels mode
   Provide --pcap and --labels where labels is a CSV mapping flow_id
   strings to labels. flow_ids are constructed deterministically by
   flow_builder._make_flow_id, so any external tooling that produces
   them will work as long as the format matches.

Train/test splitting
────────────────────
Two strategies are supported via --split:
  - random  : stratified random split (sklearn default; risks temporal leakage)
  - time    : chronological split — first 75% of flows by timestamp goes to
              train, last 25% to test. Recommended for production-realistic
              evaluation.

The CSV mode requires a `start_time` column for time-based splitting; if
absent, train.py falls back to random and warns.

Examples
────────
    # Train RF on a pre-extracted CIC-IDS CSV with chronological split
    python scripts/train.py \\
        --csv data/processed/cic_ids_2017.csv \\
        --model models/rf_v1.joblib \\
        --split time \\
        --dataset-name "CIC-IDS-2017"

    # Train XGBoost instead of RF
    python scripts/train.py \\
        --csv data/processed/ctu13.csv \\
        --model models/xgb_v1.joblib \\
        --estimator xgboost

    # Train from raw PCAP + external labels file
    python scripts/train.py \\
        --pcap data/raw/lab_capture.pcap \\
        --labels data/raw/lab_labels.csv \\
        --model models/lab_v1.joblib

Exit codes
──────────
    0 — training completed and bundle saved
    1 — runtime error (bad data, missing columns, etc.)
    2 — invalid CLI arguments
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Allow `python scripts/train.py` to find sibling modules in the package root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c2classifier.parser       import load_pcap
from c2classifier.flow_builder import FlowBuilder
from c2classifier.features     import extract, to_array, FEATURE_NAMES, N_FEATURES
from c2classifier.model        import C2Classifier

logger = logging.getLogger("train")

# Label normalisation map — accept multiple common encodings
LABEL_MAP = {
    "benign": 0, "0": 0, "normal": 0, "background": 0,
    "c2":     1, "1": 1, "malicious": 1, "botnet": 1, "attack": 1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Train a C2 traffic classifier from labeled flow data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input source ─────────────────────────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--csv",
        type=Path,
        help="Path to a pre-extracted feature CSV with FEATURE_NAMES columns + 'label'.",
    )
    src.add_argument(
        "--pcap",
        type=Path,
        help="Path to a PCAP file (used with --labels for label assignment).",
    )

    p.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="CSV with flow_id,label columns. Required when --pcap is used.",
    )

    # ── Model output ─────────────────────────────────────────────────────────
    p.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Output path for the serialised model bundle (.joblib).",
    )

    # ── Estimator selection ──────────────────────────────────────────────────
    p.add_argument(
        "--estimator",
        choices=["rf", "xgboost"],
        default="rf",
        help="Estimator family. Default: rf (Random Forest).",
    )

    # ── Training behaviour ───────────────────────────────────────────────────
    p.add_argument(
        "--split",
        choices=["random", "time"],
        default="time",
        help="Train/test split strategy. Default: time (recommended for realism).",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of data reserved for testing. Default: 0.25.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default="unspecified",
        help="Human-readable dataset name recorded in the model metadata.",
    )
    p.add_argument(
        "--min-packets",
        type=int,
        default=2,
        help="Minimum packets per flow when training from PCAP. Default: 2.",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity. -v = INFO, -vv = DEBUG.",
    )

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_from_csv(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load a pre-extracted feature CSV.

    Returns
    ───────
    X            : (n_samples, N_FEATURES) feature matrix
    y            : (n_samples,) integer labels
    start_times  : (n_samples,) flow start timestamps if 'start_time' column
                   is present; None otherwise
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    logger.info("loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Validate label column
    if "label" not in df.columns:
        raise ValueError(
            "CSV is missing required 'label' column. "
            "Expected values: benign/c2 or 0/1."
        )

    # Validate feature columns
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing {len(missing)} feature columns: {missing[:5]}"
            + ("..." if len(missing) > 5 else "")
        )

    # Normalise labels
    y_raw = df["label"].astype(str).str.lower().str.strip()
    unknown = sorted(set(y_raw) - set(LABEL_MAP.keys()))
    if unknown:
        raise ValueError(
            f"unrecognised label values: {unknown}. "
            f"Expected one of: {sorted(LABEL_MAP.keys())}"
        )
    y = y_raw.map(LABEL_MAP).to_numpy(dtype=np.int64)

    # Build feature matrix in canonical column order
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float32)

    # Replace any infs/NaNs that snuck in (CIC-IDS-2017 has known infs in
    # 'flow_bytes_per_sec' on zero-duration flows)
    bad_mask = ~np.isfinite(X).all(axis=1)
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        logger.warning("dropping %d rows with non-finite features", n_bad)
        X = X[~bad_mask]
        y = y[~bad_mask]

    start_times = None
    if "start_time" in df.columns:
        start_times = df.loc[~bad_mask, "start_time"].to_numpy(dtype=np.float64) \
                      if bad_mask.any() else df["start_time"].to_numpy(dtype=np.float64)

    logger.info("loaded %d samples (%d benign, %d c2), %d features",
                len(y), int((y == 0).sum()), int((y == 1).sum()), N_FEATURES)
    return X, y, start_times


def load_from_pcap(
    pcap_path: Path,
    labels_path: Path,
    min_packets: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build flows from a PCAP and join them against an external labels CSV.

    The labels CSV must have at minimum two columns: flow_id and label.
    flow_id values must match those produced by FlowBuilder, i.e.
    "<lo_ip>:<lo_port>-<hi_ip>:<hi_port>-<proto>".
    """
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels file not found: {labels_path}")

    logger.info("loading PCAP: %s", pcap_path)
    packets = load_pcap(pcap_path)
    if not packets:
        raise ValueError(f"no packets parsed from {pcap_path}")

    logger.info("building flows (min_packets=%d)", min_packets)
    flows = FlowBuilder(min_packets=min_packets).build(packets)
    logger.info("built %d flows", len(flows))

    # Load and normalise labels
    labels_df = pd.read_csv(labels_path)
    if not {"flow_id", "label"}.issubset(labels_df.columns):
        raise ValueError("labels CSV must contain 'flow_id' and 'label' columns")

    labels_df["label"] = labels_df["label"].astype(str).str.lower().str.strip()
    unknown = sorted(set(labels_df["label"]) - set(LABEL_MAP.keys()))
    if unknown:
        raise ValueError(f"unrecognised label values: {unknown}")
    labels_df["label_int"] = labels_df["label"].map(LABEL_MAP)

    label_lookup = dict(zip(labels_df["flow_id"], labels_df["label_int"]))

    # Match flows to labels; drop unmatched
    X_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    t_rows: List[float] = []
    n_unmatched = 0
    for flow in flows:
        if flow.flow_id not in label_lookup:
            n_unmatched += 1
            continue
        feats = extract(flow)
        X_rows.append(to_array(feats))
        y_rows.append(int(label_lookup[flow.flow_id]))
        t_rows.append(flow.start_time)

    if n_unmatched:
        logger.warning("%d flows had no matching label and were dropped", n_unmatched)
    if not X_rows:
        raise ValueError("no flows matched the labels file — check flow_id format")

    X = np.vstack(X_rows).astype(np.float32)
    y = np.array(y_rows, dtype=np.int64)
    t = np.array(t_rows, dtype=np.float64)
    logger.info("matched %d flows (%d benign, %d c2)",
                len(y), int((y == 0).sum()), int((y == 1).sum()))
    return X, y, t


# ─────────────────────────────────────────────────────────────────────────────
# Splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    start_times: Optional[np.ndarray],
    strategy: str,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split (X, y) into train/test using the requested strategy.

    Time-based split orders by start_time, then takes the last `test_size`
    fraction as the test set. This prevents the temporal leakage that
    happens when malware flows from a single attack window get scattered
    across train and test by random shuffle.
    """
    if strategy == "time":
        if start_times is None:
            logger.warning(
                "time-based split requested but no timestamps available; "
                "falling back to stratified random split"
            )
            strategy = "random"
        else:
            order = np.argsort(start_times)
            X_sorted = X[order]
            y_sorted = y[order]
            cut = int(len(y_sorted) * (1.0 - test_size))
            X_train, X_test = X_sorted[:cut], X_sorted[cut:]
            y_train, y_test = y_sorted[:cut], y_sorted[cut:]
            logger.info(
                "time-based split: train=%d (t∈[%.1f, %.1f]), test=%d (t∈[%.1f, %.1f])",
                len(y_train), start_times[order[0]], start_times[order[cut - 1]],
                len(y_test),  start_times[order[cut]], start_times[order[-1]],
            )
            # Detect single-class splits early — common when malware only
            # appears in the latter portion of a capture
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                logger.warning(
                    "time-based split produced a single-class partition. "
                    "Train classes=%s, test classes=%s. "
                    "Consider --split random or shuffling the source data.",
                    np.unique(y_train).tolist(), np.unique(y_test).tolist(),
                )
            return X_train, X_test, y_train, y_test

    # Stratified random fallback
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y if len(np.unique(y)) > 1 else None,
    )
    logger.info("random split: train=%d, test=%d", len(y_train), len(y_test))
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Estimator construction
# ─────────────────────────────────────────────────────────────────────────────

def build_estimator(name: str, random_state: int):
    """Construct the requested sklearn-compatible estimator."""
    if name == "rf":
        # Use the C2Classifier default by passing None
        return None

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install with: pip install xgboost"
            ) from exc

        return XGBClassifier(
            n_estimators       = 400,
            max_depth          = 8,
            learning_rate      = 0.05,
            subsample          = 0.9,
            colsample_bytree   = 0.9,
            objective          = "binary:logistic",
            eval_metric        = "logloss",
            tree_method        = "hist",
            n_jobs             = -1,
            random_state       = random_state,
            # scale_pos_weight is set at fit time once we know class counts
        )

    raise ValueError(f"unknown estimator: {name}")


def maybe_set_scale_pos_weight(estimator, y_train: np.ndarray) -> None:
    """For XGBoost, set scale_pos_weight from the training class distribution."""
    if estimator is None:
        return  # default RF; class_weight='balanced' already set
    cls_name = estimator.__class__.__name__
    if cls_name == "XGBClassifier":
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        if n_pos > 0:
            spw = n_neg / n_pos
            estimator.set_params(scale_pos_weight=spw)
            logger.info("xgboost scale_pos_weight set to %.3f (n_neg=%d, n_pos=%d)",
                        spw, n_neg, n_pos)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Configure logging
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

    # Validate combined-arg requirements
    if args.pcap and not args.labels:
        logger.error("--pcap requires --labels (path to flow_id,label CSV)")
        return 2

    # ── Load data ───────────────────────────────────────────────────────────
    try:
        if args.csv:
            X, y, start_times = load_from_csv(args.csv)
        else:
            X, y, start_times = load_from_pcap(
                args.pcap, args.labels, args.min_packets,
            )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("data loading failed: %s", exc)
        return 1

    if len(np.unique(y)) < 2:
        logger.error(
            "training data contains only one class (%s); cannot train a binary classifier",
            np.unique(y).tolist(),
        )
        return 1

    # ── Split ────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(
        X, y, start_times,
        strategy     = args.split,
        test_size    = args.test_size,
        random_state = args.random_state,
    )

    # ── Build classifier ─────────────────────────────────────────────────────
    estimator = build_estimator(args.estimator, args.random_state)
    maybe_set_scale_pos_weight(estimator, y_train)
    classifier = C2Classifier(
        estimator    = estimator,
        random_state = args.random_state,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    logger.info("training %s on %d samples", args.estimator, len(y_train))
    classifier.train(X_train, y_train, dataset_name=args.dataset_name)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics = classifier.evaluate(X_test, y_test, verbose=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    classifier.save(args.model)

    # ── Stdout summary ───────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"training complete — bundle saved to {args.model}")
    print(f"estimator     : {args.estimator}")
    print(f"dataset       : {args.dataset_name}")
    print(f"split         : {args.split} (test_size={args.test_size})")
    print(f"train/test    : {len(y_train)} / {len(y_test)}")
    print(f"class balance : benign={int((y_train == 0).sum())} "
          f"c2={int((y_train == 1).sum())}")
    print(f"\nmetrics:")
    for k, v in metrics.items():
        print(f"  {k:<10} {v:.4f}")
    print("─" * 60)

    # Surface feature-importance ranking when available — this is one of the
    # most useful artefacts for a portfolio writeup
    fi = getattr(classifier.model, "feature_importances_", None)
    if fi is not None:
        ranked = sorted(zip(FEATURE_NAMES, fi), key=lambda kv: kv[1], reverse=True)
        print("\ntop 10 features by importance:")
        for name, imp in ranked[:10]:
            print(f"  {name:<26} {imp:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())