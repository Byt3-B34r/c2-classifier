"""
parser.py
─────────
PCAP and live-capture ingestion for the C2 classifier.

Reads packet captures using Scapy and emits a stream (or list) of normalised
Packet records that flow_builder.py can consume directly. Supports:

    - Offline PCAP / PCAPNG files
    - Live capture from a network interface (requires elevated privileges)
    - Streaming iteration (memory-friendly for large captures)
    - BPF-style filter expressions for live capture

Design notes
────────────
- Scapy is used over pyshark/tshark to avoid the subprocess + JSON-bridge
  overhead. For 1M+ packet captures the difference is significant.
- Packet records are decoupled from scapy's layer objects — once parse_packet()
  returns, no scapy state is retained. This keeps memory bounded.
- Malformed or non-IP packets are silently skipped with a counter — captures
  from real networks always contain noise (LLC, ARP, malformed frames).
- The streaming variant (iter_pcap) is the primary entrypoint; load_pcap()
  is a convenience wrapper that materialises the full list.

Usage
─────
    from parser import load_pcap, iter_pcap, live_capture

    # Offline, materialised
    packets = load_pcap("captures/sample.pcap")

    # Offline, streaming (preferred for large files)
    for pkt in iter_pcap("captures/large.pcap"):
        ...

    # Live capture
    for pkt in live_capture("eth0", bpf_filter="tcp or udp", count=1000):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Union

# Suppress scapy's noisy import warnings before importing it
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="scapy")

try:
    from scapy.all import (
        PcapReader,
        sniff,
        IP,
        IPv6,
        TCP,
        UDP,
        ICMP,
        Raw,
    )
except ImportError as exc:
    raise ImportError(
        "scapy is required. Install with: pip install scapy"
    ) from exc

from .flow_builder import (
    Packet,
    PROTO_TCP,
    PROTO_UDP,
    PROTO_ICMP,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Parser statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParseStats:
    """Counters tracked during parsing — useful for diagnostics."""
    total_seen:   int = 0       # every frame the dissector touched
    parsed:       int = 0       # successfully converted to Packet
    skipped_l2:   int = 0       # non-IP frames (ARP, LLDP, malformed L2)
    skipped_l4:   int = 0       # non-TCP/UDP/ICMP IP packets
    skipped_err:  int = 0       # exceptions during parsing

    def summary(self) -> str:
        return (
            f"parsed={self.parsed} "
            f"skipped_l2={self.skipped_l2} "
            f"skipped_l4={self.skipped_l4} "
            f"errors={self.skipped_err} "
            f"total={self.total_seen}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_pcap(
    path: Union[str, Path],
    max_packets: Optional[int] = None,
) -> List[Packet]:
    """
    Read a PCAP file and return a sorted list of Packet records.

    Parameters
    ──────────
    path        : path to .pcap or .pcapng
    max_packets : optional cap on number of parsed packets (for quick tests)

    Returns
    ───────
    List[Packet] sorted by timestamp ascending.
    """
    packets = list(iter_pcap(path, max_packets=max_packets))
    packets.sort(key=lambda p: p.timestamp)
    return packets


def iter_pcap(
    path: Union[str, Path],
    max_packets: Optional[int] = None,
) -> Iterator[Packet]:
    """
    Stream Packet records from a PCAP file one at a time.

    Use this instead of load_pcap() when working with multi-GB captures
    where holding every packet in memory is not feasible.

    Note: order is preserved as written in the file. Most capture tools
    write packets in arrival order, but if you need guaranteed sort use
    load_pcap() which does a final sort pass.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP not found: {path}")
    if not path.is_file():
        raise ValueError(f"PCAP path is not a file: {path}")

    stats = ParseStats()
    logger.info("parser: opening PCAP %s", path)

    try:
        with PcapReader(str(path)) as reader:
            for raw in reader:
                stats.total_seen += 1

                if max_packets is not None and stats.parsed >= max_packets:
                    break

                pkt = _parse_packet(raw, stats)
                if pkt is not None:
                    stats.parsed += 1
                    yield pkt
    except Exception as exc:
        logger.error("parser: failed to read %s: %s", path, exc)
        raise
    finally:
        logger.info("parser: %s — %s", path.name, stats.summary())


def live_capture(
    interface: str,
    bpf_filter: Optional[str] = None,
    count: int = 0,
    timeout: Optional[float] = None,
) -> Iterator[Packet]:
    """
    Capture packets from a live network interface.

    Requires elevated privileges (root / CAP_NET_RAW on Linux).

    Parameters
    ──────────
    interface  : interface name (e.g. "eth0", "en0", "any")
    bpf_filter : optional BPF expression (e.g. "tcp port 443")
    count      : stop after this many packets; 0 means unlimited
    timeout    : stop after this many seconds; None means no timeout

    Yields
    ──────
    Packet records as they arrive.
    """
    stats = ParseStats()
    buffer: List[Packet] = []

    logger.info(
        "parser: live capture on %s (filter=%r, count=%d, timeout=%s)",
        interface, bpf_filter, count, timeout,
    )

    def _on_packet(raw) -> None:
        stats.total_seen += 1
        pkt = _parse_packet(raw, stats)
        if pkt is not None:
            stats.parsed += 1
            buffer.append(pkt)

    try:
        sniff(
            iface          = interface,
            filter         = bpf_filter,
            prn            = _on_packet,
            count          = count,
            timeout        = timeout,
            store          = False,   # critical: don't retain raw scapy objects
        )
    except PermissionError as exc:
        raise PermissionError(
            "live capture requires elevated privileges "
            "(run as root or grant CAP_NET_RAW)"
        ) from exc
    finally:
        logger.info("parser: live capture ended — %s", stats.summary())

    yield from buffer


# ─────────────────────────────────────────────────────────────────────────────
# Internal: scapy → Packet conversion
# ─────────────────────────────────────────────────────────────────────────────

def _parse_packet(raw, stats: ParseStats) -> Optional[Packet]:
    """
    Convert a scapy frame to a Packet record.

    Returns None — and bumps the appropriate stats counter — for frames
    that are not IPv4/IPv6 or carry an unsupported transport protocol.
    """
    try:
        # Locate the IP layer (v4 or v6)
        if IP in raw:
            ip_layer = raw[IP]
            src_ip   = ip_layer.src
            dst_ip   = ip_layer.dst
            ip_proto = ip_layer.proto
            ip_len   = int(ip_layer.len) if hasattr(ip_layer, "len") else len(raw)
        elif IPv6 in raw:
            ip_layer = raw[IPv6]
            src_ip   = ip_layer.src
            dst_ip   = ip_layer.dst
            ip_proto = ip_layer.nh           # next-header on IPv6
            ip_len   = int(ip_layer.plen) + 40 if hasattr(ip_layer, "plen") else len(raw)
        else:
            stats.skipped_l2 += 1
            return None

        # Timestamp: scapy stores it on the frame as raw.time (EDecimal in
        # newer versions; cast to float for downstream consumers)
        ts = float(getattr(raw, "time", 0.0))

        # Default values for non-port protocols
        src_port  = 0
        dst_port  = 0
        tcp_flags = 0
        payload   = b""
        payload_len = 0

        if ip_proto == PROTO_TCP and TCP in raw:
            tcp = raw[TCP]
            src_port  = int(tcp.sport)
            dst_port  = int(tcp.dport)
            tcp_flags = int(tcp.flags)
            payload   = bytes(tcp.payload) if tcp.payload else b""
            payload_len = len(payload)

        elif ip_proto == PROTO_UDP and UDP in raw:
            udp = raw[UDP]
            src_port  = int(udp.sport)
            dst_port  = int(udp.dport)
            payload   = bytes(udp.payload) if udp.payload else b""
            payload_len = len(payload)

        elif ip_proto == PROTO_ICMP and ICMP in raw:
            # ICMP has no port concept — leave ports at 0.
            # Use the ICMP type/code as a weak surrogate if needed downstream.
            payload = bytes(raw[ICMP].payload) if raw[ICMP].payload else b""
            payload_len = len(payload)

        else:
            # IP packet but transport is something we don't track
            # (e.g. GRE, ESP, IGMP). Skip cleanly.
            stats.skipped_l4 += 1
            return None

        # Optional: fall back to Raw layer for payload if transport had none
        # but scapy parsed a Raw blob (happens with truncated captures)
        if payload_len == 0 and Raw in raw:
            payload = bytes(raw[Raw].load)
            payload_len = len(payload)

        return Packet(
            timestamp   = ts,
            src_ip      = str(src_ip),
            dst_ip      = str(dst_ip),
            src_port    = src_port,
            dst_port    = dst_port,
            protocol    = int(ip_proto),
            length      = int(ip_len),
            payload_len = int(payload_len),
            tcp_flags   = tcp_flags,
            payload     = payload,
        )

    except Exception as exc:
        stats.skipped_err += 1
        logger.debug("parser: error parsing frame: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run directly: python parser.py [optional pcap path])
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # If a path was supplied, parse it; otherwise synthesise a tiny PCAP
    if len(sys.argv) > 1:
        pcap_path = sys.argv[1]
        packets = load_pcap(pcap_path, max_packets=1000)
        print(f"parsed {len(packets)} packets from {pcap_path}")
        if packets:
            print("first packet:")
            p = packets[0]
            print(f"  ts={p.timestamp:.6f}  "
                  f"{p.src_ip}:{p.src_port} -> {p.dst_ip}:{p.dst_port}  "
                  f"proto={p.protocol}  len={p.length}  payload={p.payload_len}")
        sys.exit(0)

    # Self-test path: write a synthetic PCAP and parse it back
    from scapy.all import Ether, wrpcap

    syn = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(
        sport=54321, dport=443, flags="S",
    )
    syn.time = 1_000_000.0

    syn_ack = Ether() / IP(src="10.0.0.2", dst="10.0.0.1") / TCP(
        sport=443, dport=54321, flags="SA",
    )
    syn_ack.time = 1_000_000.05

    data = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(
        sport=54321, dport=443, flags="A",
    ) / Raw(load=b"GET / HTTP/1.1\r\nHost: 10.0.0.2\r\n\r\n")
    data.time = 1_000_000.10

    arp_noise = Ether() / Raw(load=b"\x00" * 28)  # garbage non-IP frame
    arp_noise.time = 1_000_000.15

    udp = Ether() / IP(src="10.0.0.1", dst="8.8.8.8") / UDP(
        sport=53000, dport=53,
    ) / Raw(load=b"\x12\x34" + b"\x01\x00" + b"\x00\x01" * 4 + b"\x07example\x03com\x00")
    udp.time = 1_000_000.20

    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
        wrpcap(tmp.name, [syn, syn_ack, data, arp_noise, udp])
        path = tmp.name

    packets = load_pcap(path)

    assert len(packets) == 4,         f"expected 4 packets, got {len(packets)}"
    assert packets[0].protocol == PROTO_TCP
    assert packets[0].tcp_flags & 0x02, "first packet should carry SYN"
    assert packets[3].protocol == PROTO_UDP
    assert packets[3].dst_port == 53
    assert packets[2].payload_len > 0, "HTTP packet should have payload"

    print("smoke-test passed")
    print(f"\nparsed {len(packets)} packets from synthetic PCAP")
    for p in packets:
        proto_name = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(p.protocol, str(p.protocol))
        print(f"  ts={p.timestamp:.3f}  {p.src_ip}:{p.src_port:<6} -> "
              f"{p.dst_ip}:{p.dst_port:<6} {proto_name:<4} "
              f"len={p.length:<4} payload={p.payload_len:<4} flags=0x{p.tcp_flags:02x}")