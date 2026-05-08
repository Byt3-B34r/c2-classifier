# c2-classifier

A command-line tool for detecting Command & Control (C2) traffic in network captures using machine learning. Extracts bidirectional flow features from raw PCAPs, classifies flows using a trained Random Forest / XGBoost model, and produces a ranked JSON report with per-prediction SHAP explainability.

Built for security researchers, detection engineers, and SOC analysts who need a transparent, auditable alternative to black-box network detection.

---

## Features

- **PCAP ingestion** — offline PCAP files or live interface capture via Scapy / pyshark
- **Bidirectional flow reconstruction** — 5-tuple flow grouping with configurable idle timeouts
- **~30 flow-level features** — packet stats, byte ratios, IAT distribution, payload entropy, beacon regularity score, protocol metadata
- **ML classification** — Random Forest baseline; XGBoost and LSTM (beaconing) in v2
- **SHAP explainability** — per-flow feature attribution included in every report
- **Structured output** — JSON report ranked by confidence; optional CSV export
- **Trained on real data** — CIC-IDS-2017, CTU-13, and malware-traffic-analysis.net PCAPs

---

## Quick Start

### Requirements

- Python 3.10+
- `tshark` installed and on `$PATH` (for pyshark dissection)

```bash
git clone https://gitlab.com/<your-handle>/c2-classifier.git
cd c2-classifier
pip install -r requirements.txt
```

### Classify a PCAP

```bash
python scripts/classify.py --pcap captures/sample.pcap --output results.json
```

### Train a new model

```bash
python scripts/train.py --data data/processed/cic_ids_2017.csv --model models/rf_v1.joblib
```

---

## Project Structure

```
c2-classifier/
├── README.md
├── LICENSE
├── requirements.txt
├── data/
│   ├── raw/                  # PCAPs (gitignored)
│   └── processed/            # extracted flow CSVs
├── c2classifier/
│   ├── __init__.py
│   ├── parser.py             # PCAP → raw packet list
│   ├── flow_builder.py       # packets → bidirectional flows
│   ├── features.py           # flows → feature vectors
│   ├── model.py              # train / load / predict
│   └── report.py             # JSON / CSV output formatting
├── scripts/
│   ├── train.py              # offline training pipeline
│   └── classify.py           # inference entrypoint (CLI)
├── models/
│   └── rf_v1.joblib          # serialized trained model
├── notebooks/
│   └── eda.ipynb             # exploratory data analysis
└── tests/
    └── test_features.py
```

---

## Feature Vector

Each bidirectional flow is represented by ~30 features across four groups:

**Flow statistics**
- Forward / backward packet count and total bytes
- Forward / backward byte ratio
- Mean, std, min, max packet length
- Flow duration

**Timing & beaconing**
- Inter-arrival time (IAT): mean, std, min, max, skewness, kurtosis
- Beacon regularity score (coefficient of variation of IAT — low CV indicates periodic beaconing)
- Active time, idle time

**Entropy & payload**
- Payload byte entropy (Shannon)
- Header field entropy
- DNS query string entropy (where applicable)

**Protocol metadata**
- Protocol number, destination port, TCP flags bitmap
- Flow direction dominance (% of bytes in dominant direction)
- Packets-per-second, bytes-per-second, small packet ratio (<64 bytes)

---

## Training Data

| Dataset | Description | Source |
|---|---|---|
| CIC-IDS-2017 | Pre-extracted labeled flow CSVs, benign + attack traffic | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) |
| CTU-13 | Real botnet PCAPs with labeled flows (13 scenarios) | [Stratosphere IPS](https://www.stratosphereips.org/datasets-ctu13) |
| MTA-net | Real malware C2 PCAPs by family (Emotet, Cobalt Strike, etc.) | [malware-traffic-analysis.net](https://malware-traffic-analysis.net) |
| Synthetic lab | Self-generated Sliver / Havoc C2 sessions in isolated VM | Local |

> **Note:** Raw PCAPs are not included in this repository. The `data/raw/` directory is gitignored. See each dataset source for download and licensing terms.

---

## Output Format

```json
{
  "pcap": "captures/sample.pcap",
  "analyzed_at": "2025-10-15T14:32:00Z",
  "total_flows": 1847,
  "flagged_flows": 12,
  "flows": [
    {
      "flow_id": "192.168.1.42:49231-93.184.216.34:443-TCP",
      "label": "C2",
      "confidence": 0.94,
      "duration_s": 3612.4,
      "beacon_regularity": 0.03,
      "payload_entropy": 7.82,
      "shap": {
        "beacon_regularity": 0.41,
        "iat_cv": 0.38,
        "payload_entropy": 0.09,
        "fwd_bwd_byte_ratio": 0.06
      }
    }
  ]
}
```

Each flagged flow includes a `shap` block showing which features drove the classification — making detections auditable and actionable.

---

## Model Performance

Evaluated on a held-out time-window split of CIC-IDS-2017 (not a random shuffle, to prevent temporal leakage):

| Model | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | — |

> Benchmarks will be populated after initial training run. See `notebooks/eda.ipynb` for methodology.

---

## Implementation Notes

**Flow timeout policy** — TCP idle timeout: 120s. UDP idle timeout: 60s. Matches CIC-IDS-2017 conventions for feature comparability across datasets.

**Bidirectional flow key** — defined as `(min(src,dst), min(sport,dport), max(src,dst), max(sport,dport), proto)` so forward and reverse packets map to the same flow entry.

**Class imbalance** — C2 flows are rare in real captures. Training uses `class_weight='balanced'` (sklearn) or scale_pos_weight (XGBoost). Class distribution is logged at train time.

**Temporal leakage** — dataset splits are done by time window, not random shuffle, to reflect real deployment conditions.

---

## Roadmap

- [x] Project scaffolding and README
- [ ] `flow_builder.py` — bidirectional flow reconstruction
- [ ] `features.py` — full feature vector extraction
- [ ] `train.py` — training pipeline with CIC-IDS-2017
- [ ] `classify.py` — inference CLI
- [ ] SHAP output in reports
- [ ] JA3 / JA3S TLS fingerprinting (v2)
- [ ] LSTM beaconing model (v2)
- [ ] Live capture mode (v2)
- [ ] Sigma rule export from high-confidence detections (v2)

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

You are free to use, modify, and distribute this software. You may not use contributors' patent rights against them. If you distribute a modified version, you must state the changes made. See the full license text for terms.

---

## Author

[Byt3-B34r](https://github.com/Byt3-B34r) · Security researcher · Offensive + defensive tooling

> Built for research and authorized lab environments. See [responsible use](#) for scope.
