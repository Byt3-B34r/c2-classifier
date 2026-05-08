#!/usr/bin/env python3
"""
classify.py
───────────
CLI entrypoint for C2 classification on PCAP files or live interfaces.

Loads a trained model bundle, ingests packets via parser.py, builds flows
via flow_builder.py, extracts features via features.py, runs inference via
model.py, and emits results via report.py.

Examples
────────
    # Offline PCAP, JSON output, with SHAP
    python scripts/classify.py \\
        --pcap captures/sample.pcap \\
        --model models/rf_v1.joblib \\
        --output results.json

    # Offline PCAP, no SHAP (faster), CSV alongside JSON
    python scripts/classify.py \\
        --pcap captures/large.pcap \\
        --model models/rf_v1.joblib \\
        --output results.json --csv results.csv \\
        --no-shap

    # Live capture (requires elevated privileges)
    sudo python scripts/classify.py \\
        --interface eth0 \\
        --model models/rf_v1.joblib \\
        --output live.json \\
        --bpf "tcp or udp" \\
        --duration 300

    # Quick triage: only show flagged flows in stdout, suppress JSON
    python scripts/classify.py \\
        --pcap captures/sample.pcap \\
        --model models/rf_v1.joblib \\
        --quiet --threshold 0.7

Exit codes
──────────
    0 — completed successfully (regardless of detections)
    1 — runtime error (parse failure, missing model, etc.)
    2 — invalid CLI arguments
    3 — at least one flow flagged at or above --fail-threshold
        (useful for CI pipelines that should fail on detections)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Allow `python scripts/classify.py` to find sibling modules in the package root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from c2classifier.parser       import load_pcap, live_capture
from c2classifier.flow_builder import FlowBuilder, Flow
from c2classifier.features     import extract, to_array, N_FEATURES
from c2classifier.model        import C2Classifier
from c2classifier.report       import Report

logger = logging.getLogger("classify")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="classify.py",
        description="Classify network flows in a PCAP or live capture as benign or C2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input source (mutually exclusive) ────────────────────────────────────
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--pcap",
        type=Path,
        help="Path to a PCAP or PCAPNG file to analyse.",
    )
    src.add_argument(
        "--interface", "-i",
        type=str,
        help="Live capture interface name (e.g. eth0). Requires elevated privileges.",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path to the trained model bundle (.joblib).",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    p.add_argument(
        "--output", "-o",
        type=Path,
        help="Path to write the JSON report. If omitted, JSON is not written.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        help="Optional path to write a flat CSV report alongside the JSON.",
    )
    p.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress the stdout summary table.",
    )

    # ── Inference behaviour ──────────────────────────────────────────────────
    p.add_argument(
        "--no-shap",
        action="store_true",
        help="Disable SHAP explainability (faster, smaller report).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for considering a flow flagged. "
             "Affects stdout summary only. Default: 0.5.",
    )
    p.add_argument(
        "--fail-threshold",
        type=float,
        default=None,
        help="If set, exit with code 3 when any flow's C2 probability "
             "exceeds this value. Useful for CI / pipeline gating.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of flows to display in the stdout summary table. Default: 10.",
    )
    p.add_argument(
        "--min-packets",
        type=int,
        default=2,
        help="Discard flows with fewer than this many packets. Default: 2.",
    )

    # ── Live capture options ─────────────────────────────────────────────────
    live = p.add_argument_group("live capture")
    live.add_argument(
        "--bpf",
        type=str,
        default=None,
        help="BPF filter expression for live capture (e.g. 'tcp port 443').",
    )
    live.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop live capture after this many seconds.",
    )
    live.add_argument(
        "--count",
        type=int,
        default=0,
        help="Stop live capture after this many packets. 0 = unlimited.",
    )

    # ── PCAP options ─────────────────────────────────────────────────────────
    pcap_g = p.add_argument_group("pcap")
    pcap_g.add_argument(
        "--max-packets",
        type=int,
        default=None,
        help="Cap the number of packets parsed from the PCAP (debugging).",
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
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────

def _ingest(args: argparse.Namespace) -> List:
    """Read packets from the chosen source. Returns a sorted list."""
    if args.pcap:
        if not args.pcap.exists():
            logger.error("PCAP file not found: %s", args.pcap)
            sys.exit(1)
        logger.info("ingesting PCAP: %s", args.pcap)
        return load_pcap(args.pcap, max_packets=args.max_packets)

    # Live capture
    logger.info("starting live capture on %s", args.interface)
    return list(live_capture(
        interface  = args.interface,
        bpf_filter = args.bpf,
        count      = args.count,
        timeout    = args.duration,
    ))


def _build_flows(packets: List, min_packets: int) -> List[Flow]:
    builder = FlowBuilder(min_packets=min_packets)
    flows = builder.build(packets)
    logger.info("built %d flows (stats=%s)", len(flows), builder.stats)
    return flows


def _build_feature_matrix(flows: List[Flow]) -> np.ndarray:
    """Stack per-flow feature vectors into a 2-D array for batch inference."""
    if not flows:
        return np.empty((0, N_FEATURES), dtype=np.float32)
    rows = [to_array(extract(f)) for f in flows]
    return np.vstack(rows)


def _load_model(path: Path) -> C2Classifier:
    if not path.exists():
        logger.error("model bundle not found: %s", path)
        sys.exit(1)
    return C2Classifier.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Configure logging based on -v count
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

    # ── Stage 1: model ──────────────────────────────────────────────────────
    classifier = _load_model(args.model)

    # ── Stage 2: ingest ─────────────────────────────────────────────────────
    try:
        packets = _ingest(args)
    except PermissionError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.exception("ingestion failed: %s", exc)
        return 1

    if not packets:
        logger.warning("no packets ingested — nothing to classify")
        return 0

    # ── Stage 3: flow construction ───────────────────────────────────────────
    flows = _build_flows(packets, min_packets=args.min_packets)
    if not flows:
        logger.warning("no flows reconstructed — nothing to classify")
        return 0

    # ── Stage 4: feature extraction ──────────────────────────────────────────
    X = _build_feature_matrix(flows)
    logger.info("feature matrix shape: %s", X.shape)

    # ── Stage 5: inference ───────────────────────────────────────────────────
    explain = not args.no_shap
    try:
        predictions = classifier.predict(X, explain=explain)
    except ImportError as exc:
        # SHAP missing — fall back gracefully rather than failing
        logger.warning("SHAP unavailable (%s); retrying without explainability", exc)
        predictions = classifier.predict(X, explain=False)

    # ── Stage 6: report assembly ─────────────────────────────────────────────
    source = str(args.pcap) if args.pcap else f"live:{args.interface}"
    report = Report(pcap_path=source, model_metadata=classifier.metadata)
    for flow, pred in zip(flows, predictions):
        report.add(flow, pred)

    if args.output:
        report.write_json(args.output)
    if args.csv:
        report.write_csv(args.csv)

    if not args.quiet:
        print(report.summary(top=args.top))

    # ── Exit code policy ─────────────────────────────────────────────────────
    if args.fail_threshold is not None:
        max_proba = max((p.proba_c2 for p in predictions), default=0.0)
        if max_proba >= args.fail_threshold:
            logger.warning(
                "fail-threshold tripped: max p(c2)=%.3f ≥ %.3f",
                max_proba, args.fail_threshold,
            )
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())