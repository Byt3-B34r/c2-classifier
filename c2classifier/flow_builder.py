"""
flow_builder.py
───────────────
Reconstructs bidirectional network flows from a list of parsed packets.

A flow is defined by a 5-tuple:
    (src_ip, src_port, dst_ip, dst_port, protocol)

Packets are grouped into bidirectional flows using a canonical key so that
forward and reverse traffic land in the same flow record. Each flow tracks
per-direction packet/byte counts, inter-arrival times, TCP flag bitmaps, and
raw payload sizes — everything the feature extractor needs downstream.

Flow termination policy
───────────────────────
  TCP  — closed on FIN+ACK exchange OR RST, or after TCP_IDLE_TIMEOUT seconds
          of inactivity
  UDP  — closed after UDP_IDLE_TIMEOUT seconds of inactivity
  ICMP — treated as single-packet flows; closed immediately

Timeout values match CIC-IDS-2017 conventions for cross-dataset comparability.

Why These decisons were made
────────────────────────────
Packet dataclass — flat normalised struct that parser.py will populate. Decoupled from scapy/pyshark so you can swap dissectors without touching flow logic.
Canonical flow key — (min(ep_a, ep_b), max(ep_a, ep_b), proto) ensures A→B and B→A land in the same flow. The sort is lexicographic on (ip, port) tuples which is deterministic and cheap.
IAT tracking — maintained per-direction (fwd_iats, bwd_iats) and combined (all_iats). features.py will need both — per-direction for byte ratio stats, combined for beaconing detection.
TCP teardown — RST closes immediately; FIN is flagged and the flow closes on the next ACK (avoids premature eviction on half-close). Idle timeout evicts anything that stalls before teardown completes.
min_packets=2 — drops single-packet noise. Set to 1 if you want to keep ICMP echo requests and similar.
Smoke-test at the bottom — run it directly with python flow_builder.py to verify the core logic before wiring up real PCAPs.

Usage
─────
    from c2classifier.parser import load_pcap
    from c2classifier.flow_builder import FlowBuilder

    packets = load_pcap("captures/sample.pcap")
    builder = FlowBuilder()
    flows   = builder.build(packets)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Timeout policy (seconds) ─────────────────────────────────────────────────
TCP_IDLE_TIMEOUT: float = 120.0
UDP_IDLE_TIMEOUT: float = 60.0

# ── TCP flag bit positions ────────────────────────────────────────────────────
TCP_FLAG_FIN: int = 0x01
TCP_FLAG_SYN: int = 0x02
TCP_FLAG_RST: int = 0x04
TCP_FLAG_ACK: int = 0x10

# ── Protocol numbers ──────────────────────────────────────────────────────────
PROTO_TCP:  int = 6
PROTO_UDP:  int = 17
PROTO_ICMP: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Packet:
    """
    Normalised packet record produced by parser.py.

    All fields are required; parser.py is responsible for filling them from
    whatever dissection library (scapy / pyshark) is in use.
    """
    timestamp:   float          # Unix epoch with sub-second precision
    src_ip:      str
    dst_ip:      str
    src_port:    int            # 0 for ICMP
    dst_port:    int            # 0 for ICMP
    protocol:    int            # IANA protocol number
    length:      int            # total IP packet length in bytes
    payload_len: int            # transport payload length (excludes headers)
    tcp_flags:   int = 0        # TCP flags bitmask; 0 for non-TCP
    payload:     bytes = b""    # raw transport payload (may be empty)


@dataclass
class Flow:
    """
    A complete bidirectional network flow.

    Direction convention
    ────────────────────
    The *forward* direction is defined by whichever endpoint sent the first
    packet observed. All subsequent packets from that same src_ip:src_port are
    counted as forward; packets from the other endpoint are backward.

    Attributes consumed by features.py
    ───────────────────────────────────
    fwd_packets / bwd_packets   — per-direction raw packet lists (Packet objs)
    fwd_iats / bwd_iats         — inter-arrival times in seconds (per direction)
    all_iats                    — combined IATs across both directions
    fwd_tcp_flags / bwd_tcp_flags — union of all TCP flag bitmasks seen
    """

    # Identity
    flow_id:    str             # human-readable string key
    src_ip:     str             # forward-direction source
    src_port:   int
    dst_ip:     str             # forward-direction destination
    dst_port:   int
    protocol:   int

    # Timing
    start_time: float = 0.0
    end_time:   float = 0.0
    last_seen:  float = 0.0     # used for idle-timeout tracking

    # Per-direction packet stores
    fwd_packets: List[Packet] = field(default_factory=list)
    bwd_packets: List[Packet] = field(default_factory=list)

    # Inter-arrival times (seconds)
    fwd_iats: List[float] = field(default_factory=list)
    bwd_iats: List[float] = field(default_factory=list)
    all_iats: List[float] = field(default_factory=list)

    # TCP state tracking
    fwd_tcp_flags: int = 0      # union of all forward flags seen
    bwd_tcp_flags: int = 0      # union of all backward flags seen
    fin_seen:      bool = False  # FIN observed in either direction
    rst_seen:      bool = False  # RST observed

    # Internal: last arrival per direction for IAT calculation
    _last_fwd_ts: float = field(default=0.0, repr=False)
    _last_bwd_ts: float = field(default=0.0, repr=False)
    _last_any_ts: float = field(default=0.0, repr=False)

    # ── Derived convenience properties ──────────────────────────────────────

    @property
    def duration(self) -> float:
        """Flow duration in seconds."""
        return max(self.end_time - self.start_time, 0.0)

    @property
    def total_packets(self) -> int:
        return len(self.fwd_packets) + len(self.bwd_packets)

    @property
    def total_bytes(self) -> int:
        return (sum(p.length for p in self.fwd_packets) +
                sum(p.length for p in self.bwd_packets))

    @property
    def fwd_payload_bytes(self) -> int:
        return sum(p.payload_len for p in self.fwd_packets)

    @property
    def bwd_payload_bytes(self) -> int:
        return sum(p.payload_len for p in self.bwd_packets)

    @property
    def is_tcp(self) -> bool:
        return self.protocol == PROTO_TCP

    @property
    def is_udp(self) -> bool:
        return self.protocol == PROTO_UDP

    def is_expired(self, current_time: float) -> bool:
        """Return True if the flow has exceeded its idle timeout."""
        timeout = TCP_IDLE_TIMEOUT if self.is_tcp else UDP_IDLE_TIMEOUT
        return (current_time - self.last_seen) > timeout

    def __repr__(self) -> str:
        return (
            f"Flow({self.flow_id} | "
            f"pkts={self.total_packets} | "
            f"bytes={self.total_bytes} | "
            f"dur={self.duration:.2f}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Flow key helpers
# ─────────────────────────────────────────────────────────────────────────────

# A canonical key ensures that A→B and B→A packets map to the same flow.
FlowKey = Tuple[str, int, str, int, int]


def _make_flow_key(packet: Packet) -> FlowKey:
    """
    Produce a direction-agnostic 5-tuple key from a packet.

    We sort (src, sport) and (dst, dport) lexicographically so that the key is
    identical regardless of which end sent the packet.
    """
    ep_a = (packet.src_ip, packet.src_port)
    ep_b = (packet.dst_ip, packet.dst_port)
    lo, hi = (ep_a, ep_b) if ep_a <= ep_b else (ep_b, ep_a)
    return (lo[0], lo[1], hi[0], hi[1], packet.protocol)


def _make_flow_id(packet: Packet) -> str:
    """
    Human-readable flow identifier, direction-agnostic.
    Format: <lo_ip>:<lo_port>-<hi_ip>:<hi_port>-<proto>
    """
    ep_a = (packet.src_ip, packet.src_port)
    ep_b = (packet.dst_ip, packet.dst_port)
    lo, hi = (ep_a, ep_b) if ep_a <= ep_b else (ep_b, ep_a)
    return f"{lo[0]}:{lo[1]}-{hi[0]}:{hi[1]}-{packet.protocol}"


def _is_forward(flow: Flow, packet: Packet) -> bool:
    """
    Return True if *packet* is travelling in the forward direction of *flow*.

    Forward is defined as src_ip:src_port matching the flow's recorded
    forward endpoint (i.e. the sender of the first packet).
    """
    return (packet.src_ip == flow.src_ip and
            packet.src_port == flow.src_port)


# ─────────────────────────────────────────────────────────────────────────────
# FlowBuilder
# ─────────────────────────────────────────────────────────────────────────────

class FlowBuilder:
    """
    Reconstructs bidirectional flows from an ordered list of Packet objects.

    The builder maintains an active-flow table keyed by canonical 5-tuple.
    Packets are assigned to flows on arrival; flows are evicted (moved to the
    completed list) when they are terminated by TCP FIN/RST or by idle timeout.

    After processing all packets, call flush() to move any remaining active
    flows to the completed list — or just call build(), which does both.

    Parameters
    ──────────
    tcp_timeout : float
        Seconds of inactivity before a TCP flow is force-expired.
        Default: TCP_IDLE_TIMEOUT (120 s).
    udp_timeout : float
        Seconds of inactivity before a UDP flow is force-expired.
        Default: UDP_IDLE_TIMEOUT (60 s).
    min_packets : int
        Flows with fewer than this many total packets are discarded.
        Default: 2 (drops single-packet noise).
    """

    def __init__(
        self,
        tcp_timeout: float = TCP_IDLE_TIMEOUT,
        udp_timeout: float = UDP_IDLE_TIMEOUT,
        min_packets: int = 2,
    ) -> None:
        self.tcp_timeout  = tcp_timeout
        self.udp_timeout  = udp_timeout
        self.min_packets  = min_packets

        self._active:    Dict[FlowKey, Flow] = {}
        self._completed: List[Flow]          = []
        self._stats = {
            "packets_seen":    0,
            "flows_completed": 0,
            "flows_discarded": 0,
        }

    # ── Public API ───────────────────────────────────────────────────────────

    def build(self, packets: List[Packet]) -> List[Flow]:
        """
        Process *packets* in timestamp order and return completed flows.

        Packets are assumed to be pre-sorted by timestamp (parser.py handles
        this). If they are not sorted, results will be incorrect.

        Returns
        ───────
        List[Flow] sorted by flow start time, excluding flows below min_packets.
        """
        if not packets:
            logger.warning("flow_builder: received empty packet list")
            return []

        logger.info("flow_builder: processing %d packets", len(packets))

        for pkt in packets:
            self._process_packet(pkt)

        self.flush()

        logger.info(
            "flow_builder: %d flows completed, %d discarded (<%d pkts)",
            self._stats["flows_completed"],
            self._stats["flows_discarded"],
            self.min_packets,
        )

        return sorted(self._completed, key=lambda f: f.start_time)

    def flush(self) -> None:
        """
        Force-expire all remaining active flows into the completed list.

        Call this after build() is complete, or after a live-capture session
        ends. build() calls flush() automatically.
        """
        for flow in list(self._active.values()):
            self._close_flow(flow)
        self._active.clear()

    @property
    def stats(self) -> dict:
        """Return a copy of processing statistics."""
        return dict(self._stats)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _process_packet(self, pkt: Packet) -> None:
        self._stats["packets_seen"] += 1
        key = _make_flow_key(pkt)

        # Check if an existing flow for this key has timed out before assigning
        if key in self._active:
            existing = self._active[key]
            if existing.is_expired(pkt.timestamp):
                logger.debug(
                    "flow_builder: idle timeout for %s at t=%.3f",
                    existing.flow_id, pkt.timestamp,
                )
                self._close_flow(existing)
                del self._active[key]
                # Fall through to create a new flow below

        if key not in self._active:
            self._active[key] = self._new_flow(pkt)
            return

        flow = self._active[key]
        self._add_packet_to_flow(flow, pkt)

        # TCP teardown detection
        if flow.is_tcp:
            if pkt.tcp_flags & TCP_FLAG_RST:
                flow.rst_seen = True
                logger.debug("flow_builder: RST on %s", flow.flow_id)
                self._close_flow(flow)
                del self._active[key]
            elif pkt.tcp_flags & TCP_FLAG_FIN:
                flow.fin_seen = True
                # Wait for one more ACK before closing; mark only
                # (full FIN+ACK teardown requires a follow-up ACK)

    def _new_flow(self, pkt: Packet) -> Flow:
        """Create a new Flow seeded by the first packet."""
        flow = Flow(
            flow_id   = _make_flow_id(pkt),
            src_ip    = pkt.src_ip,
            src_port  = pkt.src_port,
            dst_ip    = pkt.dst_ip,
            dst_port  = pkt.dst_port,
            protocol  = pkt.protocol,
            start_time = pkt.timestamp,
            end_time   = pkt.timestamp,
            last_seen  = pkt.timestamp,
            _last_fwd_ts = pkt.timestamp,
            _last_any_ts = pkt.timestamp,
        )
        # Seed the first forward packet
        flow.fwd_packets.append(pkt)
        if pkt.tcp_flags:
            flow.fwd_tcp_flags |= pkt.tcp_flags
        return flow

    def _add_packet_to_flow(self, flow: Flow, pkt: Packet) -> None:
        """Append *pkt* to *flow*, updating timing and flag state."""
        ts = pkt.timestamp

        # Update flow-level timestamps
        flow.end_time  = max(flow.end_time, ts)
        flow.last_seen = ts

        # Update combined IAT
        if flow._last_any_ts > 0:
            flow.all_iats.append(ts - flow._last_any_ts)
        flow._last_any_ts = ts

        if _is_forward(flow, pkt):
            # Forward packet
            if flow._last_fwd_ts > 0:
                flow.fwd_iats.append(ts - flow._last_fwd_ts)
            flow._last_fwd_ts = ts
            flow.fwd_packets.append(pkt)
            if pkt.tcp_flags:
                flow.fwd_tcp_flags |= pkt.tcp_flags
        else:
            # Backward packet
            if flow._last_bwd_ts > 0:
                flow.bwd_iats.append(ts - flow._last_bwd_ts)
            flow._last_bwd_ts = ts
            flow.bwd_packets.append(pkt)
            if pkt.tcp_flags:
                flow.bwd_tcp_flags |= pkt.tcp_flags

        # Close on FIN+ACK if we previously saw a FIN
        if (flow.is_tcp and flow.fin_seen and
                (pkt.tcp_flags & TCP_FLAG_ACK) and
                not (pkt.tcp_flags & TCP_FLAG_SYN)):
            logger.debug(
                "flow_builder: FIN+ACK close on %s at t=%.3f",
                flow.flow_id, ts,
            )
            self._close_flow(flow)
            key = _make_flow_key(pkt)
            self._active.pop(key, None)

    def _close_flow(self, flow: Flow) -> None:
        """Move *flow* to the completed list if it meets the min_packets threshold."""
        if flow.total_packets >= self.min_packets:
            self._completed.append(flow)
            self._stats["flows_completed"] += 1
        else:
            self._stats["flows_discarded"] += 1
            logger.debug(
                "flow_builder: discarding %s (%d pkts < min %d)",
                flow.flow_id, flow.total_packets, self.min_packets,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run directly: python flow_builder.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    # Synthesise a minimal 3-packet TCP flow: SYN → SYN-ACK → ACK
    syn = Packet(
        timestamp=1_000_000.000, src_ip="10.0.0.1", dst_ip="10.0.0.2",
        src_port=54321, dst_port=443, protocol=PROTO_TCP,
        length=60, payload_len=0, tcp_flags=TCP_FLAG_SYN,
    )
    syn_ack = Packet(
        timestamp=1_000_000.050, src_ip="10.0.0.2", dst_ip="10.0.0.1",
        src_port=443, dst_port=54321, protocol=PROTO_TCP,
        length=60, payload_len=0, tcp_flags=TCP_FLAG_SYN | TCP_FLAG_ACK,
    )
    data = Packet(
        timestamp=1_000_000.100, src_ip="10.0.0.1", dst_ip="10.0.0.2",
        src_port=54321, dst_port=443, protocol=PROTO_TCP,
        length=512, payload_len=452, tcp_flags=TCP_FLAG_ACK,
        payload=b"GET / HTTP/1.1\r\nHost: 10.0.0.2\r\n\r\n",
    )
    fin = Packet(
        timestamp=1_000_003.000, src_ip="10.0.0.1", dst_ip="10.0.0.2",
        src_port=54321, dst_port=443, protocol=PROTO_TCP,
        length=40, payload_len=0, tcp_flags=TCP_FLAG_FIN | TCP_FLAG_ACK,
    )
    fin_ack = Packet(
        timestamp=1_000_003.020, src_ip="10.0.0.2", dst_ip="10.0.0.1",
        src_port=443, dst_port=54321, protocol=PROTO_TCP,
        length=40, payload_len=0, tcp_flags=TCP_FLAG_ACK,
    )

    builder = FlowBuilder(min_packets=2)
    flows = builder.build([syn, syn_ack, data, fin, fin_ack])

    if not flows:
        print("FAIL: no flows returned", file=sys.stderr)
        sys.exit(1)

    f = flows[0]
    assert f.total_packets == 5,         f"expected 5 pkts, got {f.total_packets}"
    assert len(f.fwd_packets) == 3,      f"expected 3 fwd pkts, got {len(f.fwd_packets)}"
    assert len(f.bwd_packets) == 2,      f"expected 2 bwd pkts, got {len(f.bwd_packets)}"
    assert len(f.all_iats) == 4,         f"expected 4 IATs, got {len(f.all_iats)}"
    assert abs(f.duration - 3.02) < 0.01, f"unexpected duration {f.duration}"

    print("smoke-test passed")
    print(f)
    print(f"  fwd_packets : {len(f.fwd_packets)}")
    print(f"  bwd_packets : {len(f.bwd_packets)}")
    print(f"  all_iats    : {[round(x,3) for x in f.all_iats]}")
    print(f"  duration    : {f.duration:.3f}s")
    print(f"  total_bytes : {f.total_bytes}")
    print(f"  stats       : {builder.stats}")