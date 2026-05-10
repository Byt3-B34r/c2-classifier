"""
features.py
───────────
Extracts a fixed-length numerical feature vector from a Flow object.

Each Flow produced by flow_builder.py is transformed into a flat dict (and
optionally a numpy array) of ~30 features across four groups:

    1. Flow statistics     — packet counts, byte totals, length distributions
    2. Timing & beaconing  — IAT statistics, beacon regularity score
    3. Entropy & payload   — Shannon entropy on payload bytes and DNS queries
    4. Protocol metadata   — protocol number, port, TCP flags, direction ratio

Feature contract
────────────────
- All values are finite floats (np.nan is never emitted; missing values → 0.0)
- Feature names are stable across versions — adding features appends to the
  end of FEATURE_NAMES so that serialised models trained on earlier vectors
  remain loadable (with the new columns zeroed out)
- Division by zero is handled explicitly throughout

Usage
─────
    from c2classifier.flow_builder import FlowBuilder
    from c2classifier.features import extract, to_array, FEATURE_NAMES

    flows   = FlowBuilder().build(packets)
    records = [extract(f) for f in flows]          # list of dicts
    X       = np.vstack([to_array(r) for r in records])  # (n_flows, n_features)
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Dict, List

import numpy as np

from .flow_builder import (
    Flow,
    PROTO_TCP,
    PROTO_UDP,
    PROTO_ICMP,
    TCP_FLAG_SYN,
    TCP_FLAG_FIN,
    TCP_FLAG_RST,
    TCP_FLAG_ACK,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature registry — ORDER IS FIXED. Append only.
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES: List[str] = [
    # ── 1. Flow statistics (14 features) ─────────────────────────────────────
    "flow_duration",            # seconds
    "total_packets",
    "total_bytes",
    "fwd_packet_count",
    "bwd_packet_count",
    "fwd_total_bytes",
    "bwd_total_bytes",
    "fwd_bwd_byte_ratio",       # fwd_bytes / (fwd_bytes + bwd_bytes); 0 if no bwd
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_min",
    "pkt_len_max",
    "small_pkt_ratio",          # fraction of packets with payload_len < 64
    "fwd_payload_bytes",
    # ── 2. Timing & beaconing (10 features) ───────────────────────────────────
    "iat_mean",                 # combined IAT
    "iat_std",
    "iat_min",
    "iat_max",
    "iat_skew",                 # Fisher skewness
    "iat_kurtosis",             # excess kurtosis
    "beacon_score",             # coefficient of variation of IAT (low → periodic)
    "pkts_per_second",
    "bytes_per_second",
    "active_time",              # sum of intra-flow active windows
    # ── 3. Entropy & payload (4 features) ─────────────────────────────────────
    "payload_entropy",          # Shannon entropy of all forward payload bytes
    "header_entropy",           # entropy over (src_port, dst_port, proto) fields
    "dns_query_entropy",        # entropy of DNS query string bytes; 0 if not DNS
    "avg_payload_size",         # mean payload byte length per forward packet
    # ── 4. Protocol metadata (6 features) ─────────────────────────────────────
    "protocol",                 # IANA protocol number
    "dst_port",
    "direction_dominance",      # max(fwd,bwd) / total_bytes — how one-sided flow is
    "fwd_syn_flag",             # 1 if SYN seen in forward direction
    "fwd_fin_flag",             # 1 if FIN seen in forward direction
    "rst_flag",                 # 1 if RST seen in either direction
]

N_FEATURES: int = len(FEATURE_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract(flow: Flow) -> Dict[str, float]:
    """
    Return a feature dict for *flow*.

    All values are guaranteed to be finite Python floats.
    Missing or undefined values (e.g. DNS entropy on a TCP flow) are 0.0.
    """
    feats: Dict[str, float] = {}

    # Gather raw lengths for reuse
    all_lengths  = [p.length      for p in flow.fwd_packets + flow.bwd_packets]
    fwd_payloads = [p.payload_len for p in flow.fwd_packets]

    feats.update(_flow_stats(flow, all_lengths, fwd_payloads))
    feats.update(_timing(flow))
    feats.update(_entropy(flow, fwd_payloads))
    feats.update(_protocol_meta(flow))

    # Sanity: ensure every registered feature is present and finite
    for name in FEATURE_NAMES:
        val = feats.get(name, 0.0)
        feats[name] = float(val) if math.isfinite(float(val)) else 0.0

    return feats


def to_array(feature_dict: Dict[str, float]) -> np.ndarray:
    """
    Convert a feature dict returned by extract() into a 1-D numpy array.

    Column order matches FEATURE_NAMES exactly so that the array can be fed
    directly into a trained sklearn / XGBoost model.
    """
    return np.array([feature_dict.get(name, 0.0) for name in FEATURE_NAMES],
                    dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Feature group implementations
# ─────────────────────────────────────────────────────────────────────────────

def _flow_stats(
    flow: Flow,
    all_lengths: List[int],
    fwd_payloads: List[int],
) -> Dict[str, float]:
    """Group 1: packet counts, byte totals, length distributions."""
    n_total   = flow.total_packets
    fwd_bytes = sum(p.length for p in flow.fwd_packets)
    bwd_bytes = sum(p.length for p in flow.bwd_packets)
    total_b   = fwd_bytes + bwd_bytes

    lengths_arr = np.array(all_lengths, dtype=np.float32) if all_lengths else np.zeros(1)
    small_count = sum(1 for p in flow.fwd_packets + flow.bwd_packets if p.payload_len < 64)

    return {
        "flow_duration":    flow.duration,
        "total_packets":    float(n_total),
        "total_bytes":      float(total_b),
        "fwd_packet_count": float(len(flow.fwd_packets)),
        "bwd_packet_count": float(len(flow.bwd_packets)),
        "fwd_total_bytes":  float(fwd_bytes),
        "bwd_total_bytes":  float(bwd_bytes),
        "fwd_bwd_byte_ratio": _safe_div(fwd_bytes, total_b),
        "pkt_len_mean":     float(np.mean(lengths_arr)),
        "pkt_len_std":      float(np.std(lengths_arr)),
        "pkt_len_min":      float(np.min(lengths_arr)),
        "pkt_len_max":      float(np.max(lengths_arr)),
        "small_pkt_ratio":  _safe_div(small_count, n_total),
        "fwd_payload_bytes": float(flow.fwd_payload_bytes),
    }


def _timing(flow: Flow) -> Dict[str, float]:
    """
    Group 2: inter-arrival time statistics and beacon regularity score.

    Beacon regularity score is the coefficient of variation (std/mean) of the
    combined IAT sequence. A low CV (< 0.1) indicates highly periodic traffic —
    the primary signature of automated C2 beaconing. A high CV indicates bursty
    or interactive traffic.
    """
    iats = flow.all_iats
    dur  = flow.duration

    if not iats:
        return {k: 0.0 for k in [
            "iat_mean", "iat_std", "iat_min", "iat_max",
            "iat_skew", "iat_kurtosis", "beacon_score",
            "pkts_per_second", "bytes_per_second", "active_time",
        ]}

    arr = np.array(iats, dtype=np.float64)
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    mn   = float(np.min(arr))
    mx   = float(np.max(arr))

    # Fisher skewness and excess kurtosis (scipy not required)
    skew = _skewness(arr)
    kurt = _kurtosis(arr)

    # Beacon regularity: CV — low = regular beaconing, high = bursty/human
    beacon_score = _safe_div(std, mean)

    # Active time: sum of IATs that are below the mean (intra-burst gaps)
    # This approximates "time the flow was actively exchanging data"
    active_time = float(np.sum(arr[arr <= mean])) if mean > 0 else 0.0

    pkts_per_sec  = _safe_div(flow.total_packets, dur)
    bytes_per_sec = _safe_div(flow.total_bytes,   dur)

    return {
        "iat_mean":        mean,
        "iat_std":         std,
        "iat_min":         mn,
        "iat_max":         mx,
        "iat_skew":        skew,
        "iat_kurtosis":    kurt,
        "beacon_score":    beacon_score,
        "pkts_per_second": pkts_per_sec,
        "bytes_per_second":bytes_per_sec,
        "active_time":     active_time,
    }


def _entropy(
    flow: Flow,
    fwd_payloads: List[int],
) -> Dict[str, float]:
    """
    Group 3: Shannon entropy over payload bytes, header fields, and DNS queries.

    High payload entropy (> 7.0) on a low-beacon-score flow is a strong
    signal for encrypted C2 (e.g. HTTPS-tunnelled Cobalt Strike, encrypted
    Sliver traffic). DNS query entropy detects DGA-generated domain names.
    """
    # Concatenate all forward payload bytes
    raw_payload = b"".join(p.payload for p in flow.fwd_packets if p.payload)
    payload_ent = _shannon_entropy(raw_payload) if raw_payload else 0.0

    # Header entropy: treat the (src_port, dst_port, protocol) triple as a
    # small byte sequence — captures unusual port / protocol distributions
    header_bytes = bytes([
        flow.src_port  & 0xFF, (flow.src_port  >> 8) & 0xFF,
        flow.dst_port  & 0xFF, (flow.dst_port  >> 8) & 0xFF,
        flow.protocol  & 0xFF,
    ])
    header_ent = _shannon_entropy(header_bytes)

    # DNS query entropy — only meaningful on port 53
    dns_ent = 0.0
    if flow.dst_port == 53 or flow.src_port == 53:
        dns_payload = b"".join(p.payload for p in flow.fwd_packets if p.payload)
        if dns_payload:
            # Strip the first 12 bytes (DNS header) if possible
            query_bytes = dns_payload[12:] if len(dns_payload) > 12 else dns_payload
            dns_ent = _shannon_entropy(query_bytes)

    avg_payload = (
        _safe_div(sum(fwd_payloads), len(fwd_payloads))
        if fwd_payloads else 0.0
    )

    return {
        "payload_entropy":   payload_ent,
        "header_entropy":    header_ent,
        "dns_query_entropy": dns_ent,
        "avg_payload_size":  avg_payload,
    }


def _protocol_meta(flow: Flow) -> Dict[str, float]:
    """Group 4: protocol number, port, TCP flags, direction dominance."""
    fwd_bytes = sum(p.length for p in flow.fwd_packets)
    bwd_bytes = sum(p.length for p in flow.bwd_packets)
    total_b   = fwd_bytes + bwd_bytes

    direction_dominance = _safe_div(max(fwd_bytes, bwd_bytes), total_b)

    fwd_syn = 1.0 if (flow.fwd_tcp_flags & TCP_FLAG_SYN) else 0.0
    fwd_fin = 1.0 if (flow.fwd_tcp_flags & TCP_FLAG_FIN) else 0.0
    rst     = 1.0 if (flow.rst_seen or
                      (flow.fwd_tcp_flags & TCP_FLAG_RST) or
                      (flow.bwd_tcp_flags & TCP_FLAG_RST)) else 0.0

    return {
        "protocol":           float(flow.protocol),
        "dst_port":           float(flow.dst_port),
        "direction_dominance":direction_dominance,
        "fwd_syn_flag":       fwd_syn,
        "fwd_fin_flag":       fwd_fin,
        "rst_flag":           rst,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(data: bytes) -> float:
    """
    Compute Shannon entropy of a byte sequence in bits per byte.

    Returns a value in [0.0, 8.0].
    High values (> 7.0) indicate near-random data — characteristic of
    encrypted or compressed payloads.
    """
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        entropy -= p * math.log2(p)
    return entropy


def _skewness(arr: np.ndarray) -> float:
    """Fisher skewness (moment-based, no scipy dependency)."""
    n = len(arr)
    if n < 3:
        return 0.0
    mean = np.mean(arr)
    std  = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    """
    Excess kurtosis (Fisher definition: normal distribution → 0).
    No scipy dependency.
    """
    n = len(arr)
    if n < 4:
        return 0.0
    mean = np.mean(arr)
    std  = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(((arr - mean) / std) ** 4)) - 3.0


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of raising ZeroDivisionError."""
    if denominator == 0:
        return 0.0
    result = numerator / denominator
    return float(result) if math.isfinite(result) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run directly: python features.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from flow_builder import FlowBuilder, Packet

    logging.basicConfig(level=logging.WARNING)

    # Build a synthetic periodic beacon flow: 10 packets, ~60s IAT (C2-like)
    pkts = []
    for i in range(10):
        direction = i % 2  # alternate fwd/bwd
        pkts.append(Packet(
            timestamp   = 1_000_000.0 + i * 60.0 + (i * 0.01),  # near-periodic
            src_ip      = "10.0.0.1"   if direction == 0 else "93.184.216.34",
            dst_ip      = "93.184.216.34" if direction == 0 else "10.0.0.1",
            src_port    = 54321        if direction == 0 else 443,
            dst_port    = 443          if direction == 0 else 54321,
            protocol    = PROTO_TCP,
            length      = 128,
            payload_len = 88,
            tcp_flags   = TCP_FLAG_ACK,
            payload     = bytes(range(88)),  # synthetic non-random payload
        ))

    flows = FlowBuilder(min_packets=2).build(pkts)
    assert flows, "smoke-test: no flows produced"

    f = flows[0]
    feats = extract(f)

    # Verify all registered features are present and finite
    missing = [k for k in FEATURE_NAMES if k not in feats]
    assert not missing, f"missing features: {missing}"

    non_finite = [k for k, v in feats.items() if not math.isfinite(v)]
    assert not non_finite, f"non-finite values: {non_finite}"

    arr = to_array(feats)
    assert arr.shape == (N_FEATURES,), f"expected shape ({N_FEATURES},), got {arr.shape}"
    assert arr.dtype == np.float32

    print(f"smoke-test passed — {N_FEATURES} features extracted")
    print(f"\nFlow:  {f}")
    print(f"\nTop features:")
    interesting = [
        "beacon_score", "payload_entropy", "iat_mean", "iat_std",
        "fwd_bwd_byte_ratio", "pkts_per_second", "flow_duration",
        "direction_dominance",
    ]
    for k in interesting:
        print(f"  {k:<26} {feats[k]:.4f}")
    print(f"\nArray shape: {arr.shape}, dtype: {arr.dtype}")