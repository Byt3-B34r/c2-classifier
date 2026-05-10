"""
report.py
─────────
Output formatting for C2 classifier results.

Takes a stream of (Flow, Prediction) pairs and produces:

    - JSON report ranked by C2 confidence (primary deliverable)
    - CSV export for spreadsheet / SIEM ingestion (optional)
    - Concise stdout summary for CLI feedback

The JSON schema is the contract advertised in the README — downstream tools
(Sigma rule generators, dashboards, SOC pipelines) parse it. Keep the schema
backward-compatible: add fields, never rename or remove them.

Usage
─────
    from report import Report

    report = Report(pcap_path="captures/sample.pcap", model_metadata=clf.metadata)
    for flow, pred in zip(flows, predictions):
        report.add(flow, pred)

    report.write_json("results.json")
    report.write_csv("results.csv")        # optional
    print(report.summary())                # stdout
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .flow_builder import Flow, PROTO_TCP, PROTO_UDP, PROTO_ICMP
from .features import extract as extract_features
from .model import Prediction, TrainingMetadata

logger = logging.getLogger(__name__)

# Top-N SHAP contributors retained per flow in the JSON output.
# More than 8 makes the report noisy; fewer hides genuinely contributing features.
SHAP_TOP_N: int = 8

# Confidence threshold for the "flagged" count in the summary.
# Flows below this are still in the report — just not counted as flagged.
FLAG_THRESHOLD: float = 0.5

# JSON schema version — bump on breaking schema changes so consumers can
# detect format drift programmatically.
SCHEMA_VERSION: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Per-flow record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowRecord:
    """
    Serialised representation of a single classified flow.

    The field order here is the JSON output order (Python dicts preserve
    insertion order since 3.7). Reorder thoughtfully — downstream parsers
    may rely on it for human-readable output.
    """
    flow_id:        str
    label:          str               # "benign" | "c2"
    confidence:     float             # confidence in the predicted label
    proba_c2:       float             # probability of c2 specifically (for ranking)

    # Five-tuple identity
    src_ip:         str
    src_port:       int
    dst_ip:         str
    dst_port:       int
    protocol:       str               # "TCP" / "UDP" / "ICMP" / numeric

    # Flow-level summary stats (the most useful subset of features)
    duration_s:         float
    total_packets:      int
    total_bytes:        int
    beacon_score:       float
    payload_entropy:    float
    pkts_per_second:    float
    iat_mean:           float

    # Optional SHAP attribution: top-N features by absolute contribution
    shap: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.shap is None:
            d.pop("shap")
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

class Report:
    """
    Accumulates classified-flow records and emits structured output.

    Parameters
    ──────────
    pcap_path : path of the analysed PCAP (or interface name for live capture)
    model_metadata : TrainingMetadata from the loaded classifier; embedded in
                     the report so consumers know which model produced it
    """

    def __init__(
        self,
        pcap_path:      Union[str, Path],
        model_metadata: Optional[TrainingMetadata] = None,
    ) -> None:
        self.source         = str(pcap_path)
        self.model_metadata = model_metadata
        self.records: List[FlowRecord] = []
        self._created_at    = datetime.now(timezone.utc).isoformat()

    # ── Accumulation ─────────────────────────────────────────────────────────

    def add(self, flow: Flow, prediction: Prediction) -> None:
        """Append a (flow, prediction) pair to the report."""
        feats = extract_features(flow)

        shap_summary: Optional[Dict[str, float]] = None
        if prediction.shap:
            # Retain only the top-N most influential features (by |contribution|)
            top = sorted(
                prediction.shap.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )[:SHAP_TOP_N]
            shap_summary = {name: round(val, 6) for name, val in top}

        record = FlowRecord(
            flow_id          = flow.flow_id,
            label            = prediction.label,
            confidence       = round(prediction.confidence, 6),
            proba_c2         = round(prediction.proba_c2,   6),
            src_ip           = flow.src_ip,
            src_port         = flow.src_port,
            dst_ip           = flow.dst_ip,
            dst_port         = flow.dst_port,
            protocol         = _proto_name(flow.protocol),
            duration_s       = round(feats["flow_duration"],    3),
            total_packets    = int(feats["total_packets"]),
            total_bytes      = int(feats["total_bytes"]),
            beacon_score     = round(feats["beacon_score"],     6),
            payload_entropy  = round(feats["payload_entropy"],  4),
            pkts_per_second  = round(feats["pkts_per_second"],  4),
            iat_mean         = round(feats["iat_mean"],         6),
            shap             = shap_summary,
        )
        self.records.append(record)

    # ── Output: JSON ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Build the full JSON-serialisable report structure."""
        flagged = [r for r in self.records if r.proba_c2 >= FLAG_THRESHOLD]

        # Sort flows by C2 probability descending so the most suspicious
        # appear first — analysts always read top-to-bottom.
        sorted_records = sorted(
            self.records,
            key=lambda r: r.proba_c2,
            reverse=True,
        )

        return {
            "schema_version":    SCHEMA_VERSION,
            "source":            self.source,
            "analyzed_at":       self._created_at,
            "total_flows":       len(self.records),
            "flagged_flows":     len(flagged),
            "flag_threshold":    FLAG_THRESHOLD,
            "model":             self._model_block(),
            "flows":             [r.to_dict() for r in sorted_records],
        }

    def write_json(
        self,
        path: Union[str, Path],
        indent: int = 2,
    ) -> None:
        """Write the report to disk as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=indent, ensure_ascii=False)
        logger.info("report: wrote JSON to %s (%d flows)", path, len(self.records))

    # ── Output: CSV ──────────────────────────────────────────────────────────

    def write_csv(self, path: Union[str, Path]) -> None:
        """
        Write a flat CSV of all flows, sorted by C2 probability descending.

        SHAP attributions are flattened into a single semicolon-delimited
        string column ('feature=value;feature=value;...'). This keeps the
        CSV ingestible by Splunk / Elastic / Excel without nested-object pain.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        sorted_records = sorted(
            self.records,
            key=lambda r: r.proba_c2,
            reverse=True,
        )

        fieldnames = [
            "flow_id", "label", "confidence", "proba_c2",
            "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
            "duration_s", "total_packets", "total_bytes",
            "beacon_score", "payload_entropy", "pkts_per_second", "iat_mean",
            "shap_top",
        ]

        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in sorted_records:
                row = r.to_dict()
                # Flatten SHAP into a single column
                shap_dict = row.pop("shap", None)
                row["shap_top"] = (
                    ";".join(f"{k}={v}" for k, v in shap_dict.items())
                    if shap_dict else ""
                )
                writer.writerow(row)

        logger.info("report: wrote CSV to %s (%d flows)", path, len(self.records))

    # ── Output: stdout summary ───────────────────────────────────────────────

    def summary(self, top: int = 10) -> str:
        """
        Return a human-readable summary suitable for stdout.

        Shows total/flagged counts and the top-N flows by C2 probability.
        """
        flagged = [r for r in self.records if r.proba_c2 >= FLAG_THRESHOLD]

        sorted_records = sorted(
            self.records,
            key=lambda r: r.proba_c2,
            reverse=True,
        )[:top]

        lines = []
        lines.append(f"source         : {self.source}")
        lines.append(f"analyzed_at    : {self._created_at}")
        lines.append(f"total_flows    : {len(self.records)}")
        lines.append(f"flagged (≥{FLAG_THRESHOLD:.2f}) : {len(flagged)}")

        if self.model_metadata and self.model_metadata.metrics:
            m = self.model_metadata.metrics
            metric_str = "  ".join(f"{k}={v:.3f}" for k, v in m.items()
                                   if k in ("precision", "recall", "f1", "fpr"))
            lines.append(f"model metrics  : {metric_str}")

        if not sorted_records:
            lines.append("\n(no flows to display)")
            return "\n".join(lines)

        lines.append(f"\nTop {min(top, len(sorted_records))} flows by C2 probability:")
        lines.append(
            f"{'flow_id':<48} "
            f"{'label':<7} "
            f"{'p(c2)':>7} "
            f"{'beacon':>8} "
            f"{'entropy':>8} "
            f"{'duration':>10}"
        )
        lines.append("─" * 92)

        for r in sorted_records:
            fid = r.flow_id[:47]
            lines.append(
                f"{fid:<48} "
                f"{r.label:<7} "
                f"{r.proba_c2:>7.3f} "
                f"{r.beacon_score:>8.4f} "
                f"{r.payload_entropy:>8.3f} "
                f"{r.duration_s:>9.2f}s"
            )

        return "\n".join(lines)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _model_block(self) -> Dict[str, Any]:
        """
        Build the `model` sub-object for the JSON report.

        Provides enough provenance for an analyst to know which classifier
        produced the result without leaking the entire training set.
        """
        if self.model_metadata is None:
            return {"trained_at": None, "dataset": None, "metrics": {}}

        return {
            "trained_at":   self.model_metadata.trained_at,
            "dataset":      self.model_metadata.dataset,
            "n_train":      self.model_metadata.n_train,
            "n_test":       self.model_metadata.n_test,
            "class_counts": self.model_metadata.class_counts,
            "metrics":      self.model_metadata.metrics,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _proto_name(proto_num: int) -> str:
    """Map IANA protocol number to human-readable string."""
    return {
        PROTO_ICMP: "ICMP",
        PROTO_TCP:  "TCP",
        PROTO_UDP:  "UDP",
    }.get(proto_num, str(proto_num))


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run directly: python report.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    from flow_builder import FlowBuilder, Packet, TCP_FLAG_ACK

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Build two synthetic flows: one beacon-like, one bursty
    beacon_pkts = []
    for i in range(20):
        direction = i % 2
        beacon_pkts.append(Packet(
            timestamp=1_000_000.0 + i * 60.0 + (i * 0.005),
            src_ip   = "10.0.0.1"      if direction == 0 else "93.184.216.34",
            dst_ip   = "93.184.216.34" if direction == 0 else "10.0.0.1",
            src_port = 54321 if direction == 0 else 443,
            dst_port = 443   if direction == 0 else 54321,
            protocol = PROTO_TCP, length=140, payload_len=100,
            tcp_flags=TCP_FLAG_ACK, payload=bytes(range(100)),
        ))

    bursty_pkts = []
    for i, gap in enumerate([0.001, 0.002, 0.001, 5.0, 0.001, 0.001, 8.0, 0.001]):
        ts = 2_000_000.0 + sum([0.001, 0.002, 0.001, 5.0, 0.001, 0.001, 8.0, 0.001][:i+1])
        direction = i % 2
        bursty_pkts.append(Packet(
            timestamp=ts,
            src_ip   = "10.0.0.5"  if direction == 0 else "1.1.1.1",
            dst_ip   = "1.1.1.1"   if direction == 0 else "10.0.0.5",
            src_port = 51000 if direction == 0 else 80,
            dst_port = 80    if direction == 0 else 51000,
            protocol = PROTO_TCP, length=512, payload_len=460,
            tcp_flags=TCP_FLAG_ACK, payload=b"GET /index HTTP/1.1\r\n" * 20,
        ))

    flows = FlowBuilder(min_packets=2).build(beacon_pkts + bursty_pkts)
    assert len(flows) == 2, f"expected 2 flows, got {len(flows)}"

    # Synthesise predictions instead of training a model
    fake_predictions = [
        Prediction(
            label      = "c2",
            confidence = 0.91,
            proba_c2   = 0.91,
            shap       = {"beacon_score": 0.42, "payload_entropy": 0.31,
                          "iat_std":     -0.18, "dst_port":         0.04},
        ),
        Prediction(
            label      = "benign",
            confidence = 0.88,
            proba_c2   = 0.12,
            shap       = {"beacon_score": -0.35, "iat_std":         0.21,
                          "payload_entropy": -0.10, "dst_port":     0.02},
        ),
    ]

    fake_meta = TrainingMetadata(
        trained_at    = "2026-01-15T10:00:00+00:00",
        dataset       = "smoke_test_synthetic",
        n_train       = 1000, n_test = 250,
        class_counts  = {"benign": 500, "c2": 500},
        metrics       = {"precision": 0.97, "recall": 0.94, "f1": 0.955, "fpr": 0.02},
    )

    report = Report(pcap_path="smoke_test.pcap", model_metadata=fake_meta)
    for flow, pred in zip(flows, fake_predictions):
        report.add(flow, pred)

    # JSON round-trip
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tmp:
        json_path = tmp.name
    report.write_json(json_path)

    with open(json_path) as fh:
        data = json.load(fh)

    assert data["schema_version"]  == SCHEMA_VERSION
    assert data["total_flows"]     == 2
    assert data["flagged_flows"]   == 1
    assert data["flows"][0]["proba_c2"] > data["flows"][1]["proba_c2"], \
        "flows must be sorted by proba_c2 descending"
    assert "shap" in data["flows"][0]
    assert data["model"]["dataset"] == "smoke_test_synthetic"

    # CSV round-trip
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        csv_path = tmp.name
    report.write_csv(csv_path)

    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert "shap_top" in rows[0]
    assert "beacon_score=" in rows[0]["shap_top"]

    print("smoke-test passed\n")
    print(report.summary(top=5))
    print(f"\nJSON output : {json_path}")
    print(f"CSV  output : {csv_path}")