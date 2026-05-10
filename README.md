# c2-classifier

A command-line tool for detecting Command & Control (C2) traffic in network captures using machine learning. Extracts bidirectional flow features from raw PCAPs, classifies flows using a trained Random Forest model, and produces a ranked JSON report with per-prediction SHAP explainability.

Built as a transparent, auditable alternative to black-box network detection — every alert ships with feature attribution, every model artefact carries its full training provenance, and the validation methodology explicitly tests cross-family generalization rather than reporting only test-set metrics.

---

## Why this exists

Most ML-based intrusion detection projects on GitHub stop at "F1 = 0.99 on CIC-IDS-2017." That number is meaningless without independent validation. This project:

1. Trains a Random Forest classifier on the CIC-IDS-2017 botnet subset
2. Validates against an independent capture from malware-traffic-analysis.net (Emotet+Trickbot)
3. Documents the gap between test-set performance and real-world generalization
4. Uses controlled feature ablation to identify *why* the gap exists
5. Plans v3 work to close the gap

Detection-engineering rigor matters more than benchmark numbers.

---

## Features

- **PCAP ingestion** — offline files or live interface capture via Scapy (streaming, memory-bounded)
- **Bidirectional flow reconstruction** — canonical 5-tuple grouping with configurable TCP/UDP idle timeouts (CIC-IDS-2017 conventions by default)
- **34 flow-level features** across four groups: flow statistics, timing & beaconing, entropy & payload, protocol metadata
- **ML classification** — Random Forest baseline; XGBoost supported via `--estimator xgboost`
- **SHAP explainability** — per-flow feature attribution included in every report
- **Imbalance-honest metrics** — PR-AUC, balanced accuracy, MCC reported alongside F1, with FPR and FNR for SOC viability
- **Feature ablation** — `--zero-features` flag for controlled experiments on artifact dependencies
- **Self-describing model bundles** — every saved model carries dataset name, training date, class counts, metrics, and feature schema

---

## Quick Start

### Requirements

- Python 3.10+
- `tshark` available on `$PATH` (optional; only needed for the pyshark fallback path)

```bash
git clone https://github.com/Byt3-B34r/c2-classifier.git
cd c2-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Classify a PCAP

```bash
python scripts/classify.py \
    --pcap captures/sample.pcap \
    --model models/rf_v1.joblib \
    --output results.json \
    --csv results.csv \
    --top 20 \
    -v
```

### Train a new model

```bash
# 1. Convert CIC-IDS-2017 ML CSVs into the project schema
python scripts/preprocess_cic.py \
    --input "data/raw/MachineLearningCSV/*.csv" \
    --output data/processed/cic_botnet.csv \
    --only-botnet \
    --benign-sample 50000 \
    -v

# 2. Train
python scripts/train.py \
    --csv data/processed/cic_botnet.csv \
    --model models/rf_v1.joblib \
    --split random \
    --dataset-name 'CIC-IDS-2017 (botnet)' \
    -v
```

---

## Project Structure

```
c2-classifier/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # PCAPs (gitignored)
│   └── processed/            # extracted flow CSVs
├── c2classifier/
│   ├── __init__.py
│   ├── parser.py             # PCAP → Packet records (Scapy streaming)
│   ├── flow_builder.py       # Packets → bidirectional Flows
│   ├── features.py           # Flows → 34-feature vectors
│   ├── model.py              # train / save / load / predict + SHAP
│   └── report.py             # JSON / CSV / stdout output
├── scripts/
│   ├── preprocess_cic.py     # CIC-IDS-2017 → project schema CSV
│   ├── train.py              # offline training pipeline
│   └── classify.py           # inference CLI
├── models/
│   └── rf_v1.joblib          # serialized trained model (gitignored)
├── notebooks/
│   └── eda.ipynb             # exploratory data analysis
└── tests/
    └── test_features.py
```

---

## Feature Vector

34 features per bidirectional flow, in fixed canonical order (see `c2classifier/features.py::FEATURE_NAMES`).

**Flow statistics (14 features)**
- Forward / backward packet count and total bytes
- Forward / backward byte ratio
- Mean, std, min, max packet length
- Flow duration, total packets, total bytes
- Small packet ratio (payload < 64 bytes)
- Forward payload bytes

**Timing & beaconing (10 features)**
- Inter-arrival time: mean, std, min, max, skewness, kurtosis
- Beacon regularity score (coefficient of variation of IAT — low CV indicates periodic beaconing)
- Packets-per-second, bytes-per-second, active time

**Entropy & payload (4 features)**
- Payload byte entropy (Shannon, bits/byte)
- Header field entropy
- DNS query string entropy (port-gated to 53)
- Average payload size

**Protocol metadata (6 features)**
- Protocol number, destination port
- Direction dominance (% of bytes in dominant direction)
- Forward SYN flag, forward FIN flag, RST flag (boolean)

> **Note on entropy and IAT-distribution features.** When training on CIC-IDS-2017 pre-extracted CSVs, four features (`payload_entropy`, `iat_skew`, `iat_kurtosis`, `dns_query_entropy`) are zero-filled because CICFlowMeter doesn't preserve raw payload or full IAT sequences. These features become populated only when training on raw PCAPs through this project's full pipeline. v3 work will exercise them.

---

## Training Data

| Dataset | Description | Source |
|---|---|---|
| CIC-IDS-2017 | Pre-extracted labeled flow CSVs (CICFlowMeter), Botnet ARES + benign | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) |
| MTA-net (Emotet+Trickbot) | Real malware C2 PCAP, used for independent validation | [malware-traffic-analysis.net](https://www.malware-traffic-analysis.net/2020/02/06/index.html) |
| CTU-13 *(planned, v3)* | 13-family botnet corpus for cross-family generalization | [Stratosphere IPS](https://www.stratosphereips.org/datasets-ctu13) |
| Synthetic lab *(planned, v3)* | Self-generated Sliver/Havoc C2 in isolated VM | Local |

Raw PCAPs are not included; `data/raw/` is gitignored. Each dataset has its own download and licensing terms.

---

## Validation Methodology and Findings

The classifier was trained on the CIC-IDS-2017 Botnet ARES subset and evaluated against an independent capture from malware-traffic-analysis.net (2020-02-06 Emotet+Trickbot). Two model variants were trained to isolate dataset artifact dependence.

### Test-set metrics (CIC-IDS-2017, 25% held-out random split)

| Model | Precision | Recall | F1 | FPR | FNR | PR-AUC | MCC |
|---|---|---|---|---|---|---|---|
| **v1** — all features | 0.867 | 0.963 | 0.913 | 0.58% | 3.68% | 0.971 | 0.911 |
| **v2** — port features ablated | 0.611 | 0.984 | 0.754 | 2.45% | 1.64% | 0.898 | 0.765 |
| **v3** — CTU-13 + raw PCAP features | *planned* | | | | | | |

### Independent validation (Emotet+Trickbot PCAP, 154 flows, 27 substantive ≥20 packets)

| Model | Flagged flows (≥0.5) | Highest p(c2) | Detection rate |
|---|---|---|---|
| **v1** | 0 / 154 | 0.145 | 0% |
| **v2** | 0 / 154 | 0.173 | 0% |

### Interpretation

**v1** achieved strong CIC test-set metrics (F1 = 0.913, MCC = 0.911, FPR = 0.58%) but failed to detect any C2 flow in the independent Trickbot capture. Top feature importance: `dst_port` (21%), `bwd_total_bytes` (10%), `avg_payload_size` (9%).

**v2** ablated `dst_port` and `protocol` to test whether port-pattern memorization was the failure cause. Result: CIC F1 dropped 17 points (0.913 → 0.754), FPR quadrupled (0.58% → 2.45%) — but Trickbot detection did not improve. This confirms that the model's reliance on Botnet-ARES-specific signatures is encoded across multiple correlated features (byte distributions, packet length statistics, flow-shape ratios), not just the destination port. Surgical feature ablation cannot remediate training-data narrowness.

**Root cause.** CIC-IDS-2017's "botnet" category contains a single malware family (Botnet ARES) with a consistent multi-dimensional signature. A model trained on it learns that signature in full and cannot generalize to families with different traffic profiles (Trickbot uses HTTPS-blending ports 80/443/8080 and different beaconing patterns). This finding is consistent with documented CIC-IDS-2017 dataset limitations (see [Rosay et al. 2022](https://lycos-ids.univ-lemans.fr/)).

**v3 plan.**
- Train on CTU-13's 13-family botnet corpus to force cross-family generalization
- Process raw PCAPs through the project's full feature pipeline so payload entropy, IAT skewness/kurtosis, and DNS query entropy are populated rather than zero-filled
- Re-validate against the same Trickbot capture plus additional MTA-net samples
- Evaluate whether MCC on cross-family validation can exceed 0.5 (the v1/v2 floor)

---

## Output Format

```json
{
  "schema_version": 1,
  "source": "captures/sample.pcap",
  "analyzed_at": "2026-05-09T14:32:00+00:00",
  "total_flows": 154,
  "flagged_flows": 0,
  "flag_threshold": 0.5,
  "model": {
    "trained_at": "2026-05-09T03:33:00+00:00",
    "dataset": "CIC-IDS-2017 (botnet)",
    "n_train": 38967,
    "n_test": 12989,
    "class_counts": {"benign": 37500, "c2": 1467},
    "metrics": {
      "precision": 0.867,
      "recall": 0.963,
      "f1": 0.913,
      "fpr": 0.0058,
      "mcc": 0.911
    }
  },
  "flows": [
    {
      "flow_id": "10.20.30.101:49852-80.86.91.91:8080-6",
      "label": "benign",
      "confidence": 0.866,
      "proba_c2": 0.134,
      "src_ip": "10.20.30.101",
      "src_port": 49852,
      "dst_ip": "80.86.91.91",
      "dst_port": 8080,
      "protocol": "TCP",
      "duration_s": 8.91,
      "total_packets": 212,
      "total_bytes": 184320,
      "beacon_score": 6.6596,
      "payload_entropy": 2.688,
      "pkts_per_second": 23.79,
      "iat_mean": 0.042,
      "shap": {
        "pkt_len_min": -0.084,
        "dst_port": -0.034,
        "fwd_total_bytes": -0.031,
        "total_bytes": -0.031
      }
    }
  ]
}
```

Each flow record includes the top SHAP contributors by absolute magnitude. Positive values pushed the classification toward c2; negative toward benign. Reports are sorted by `proba_c2` descending so analysts read the highest-suspicion flows first.

---

## Implementation Notes

**Flow timeout policy.** TCP idle timeout: 120s. UDP idle timeout: 60s. Matches CIC-IDS-2017 conventions for cross-dataset comparability.

**Bidirectional flow key.** Canonical 5-tuple `(min(ep_a, ep_b), max(ep_a, ep_b), proto)` ensures forward and reverse packets map to the same flow entry regardless of which side initiated.

**TCP teardown handling.** RST closes the flow immediately. FIN is flagged and the flow closes on the next ACK (avoids premature eviction on half-close). Idle timeout handles flows that stall before teardown completes.

**Class imbalance.** Training defaults to `class_weight='balanced'` (sklearn) or auto-derived `scale_pos_weight` (XGBoost). Class distribution is logged at train time. CIC-IDS-2017 botnet ratio is ~1:1160 raw; recommend `--benign-sample 50000` to bring it to ~25:1 for stable training.

**Imbalance-honest metrics.** ROC-AUC is reported but not relied upon. PR-AUC, balanced accuracy, MCC, and FPR/FNR are the primary evaluation metrics — ROC-AUC inflates artificially on imbalanced data.

**Feature schema is append-only.** `FEATURE_NAMES` defines the model contract; new features are appended, never inserted or removed mid-list. Saved model bundles record their schema and `load()` warns on mismatch.

**Bundle versioning.** Saved models include a `bundle_version` integer. `load()` rejects bundles newer than the running code understands; older bundles load with backward-compatible defaults.

**Temporal leakage.** Time-based train/test splitting is supported via `--split time` when timestamps are available. CIC-IDS-2017 MachineLearningCVE strips timestamps, so this dataset uses random split (documented limitation).

---

## CLI Reference

### `scripts/preprocess_cic.py`

Convert CIC-IDS-2017 CSVs into the project's feature schema.

| Flag | Purpose |
|---|---|
| `--input PATH` | Glob or single path to CIC CSV(s) |
| `--output PATH` | Output CSV |
| `--only-botnet` | Drop non-Botnet attack flows; keep BENIGN + Botnet only |
| `--benign-sample N` | Subsample BENIGN class to N rows for class balance |

### `scripts/train.py`

Train a classifier from a preprocessed CSV or raw PCAP+labels.

| Flag | Purpose |
|---|---|
| `--csv PATH` | Input feature CSV (preprocessed) |
| `--pcap PATH --labels PATH` | Alternative: raw PCAP + flow_id-to-label CSV |
| `--model PATH` | Output bundle path |
| `--estimator {rf,xgboost}` | Classifier choice; default rf |
| `--split {random,time}` | Train/test split strategy; time recommended when timestamps available |
| `--zero-features list` | Comma-separated features to zero before training (for ablation) |
| `--dataset-name STR` | Recorded in saved bundle metadata |

### `scripts/classify.py`

Run inference on a PCAP or live interface.

| Flag | Purpose |
|---|---|
| `--pcap PATH` | Offline PCAP file |
| `--interface NAME` | Live capture (requires elevated privileges) |
| `--model PATH` | Trained bundle |
| `--output PATH` | JSON report path |
| `--csv PATH` | Optional CSV report |
| `--no-shap` | Skip SHAP attribution (faster, smaller report) |
| `--fail-threshold F` | Exit code 3 when any flow's p(c2) ≥ F (CI gating) |
| `--top N` | Number of flows in stdout summary table |

---

## Roadmap

- [x] Project scaffolding, README, license, gitignore, requirements
- [x] `flow_builder.py` — bidirectional flow reconstruction with TCP teardown handling
- [x] `features.py` — 34-feature vector with append-only schema
- [x] `parser.py` — Scapy-based streaming PCAP and live capture ingestion
- [x] `model.py` — train/save/load/predict with SHAP explainability and imbalance-honest metrics
- [x] `report.py` — JSON, CSV, and stdout output with SHAP attribution
- [x] `preprocess_cic.py` — CIC-IDS-2017 ingestion
- [x] `train.py` and `classify.py` CLI entrypoints
- [x] v1 trained on CIC-IDS-2017 (Botnet ARES) — F1 = 0.913, MCC = 0.911
- [x] v1 validated against independent Emotet+Trickbot PCAP — 0/154 detection, generalization gap documented
- [x] v2 with port-feature ablation — confirmed multi-feature artifact dependence
- [ ] **v3** — train on CTU-13 multi-family botnet corpus via raw PCAP pipeline
- [ ] `tests/test_features.py` — formal test layer for feature math
- [ ] `preprocess_ctu13.py` — CTU-13 binetflow ingestion
- [ ] JA3 / JA3S TLS fingerprinting
- [ ] LSTM beaconing model for periodicity detection
- [ ] Sigma rule export from high-confidence detections

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

You are free to use, modify, and distribute this software. You may not use contributors' patent rights against them. If you distribute a modified version, you must state the changes made. See the full license text for terms.

---

## Author

[Byt3-B34r](https://github.com/Byt3-B34r) — security researcher, offensive + defensive tooling.

> Built for research and authorized lab environments. PCAPs from malware-traffic-analysis.net contain live malware binaries — handle in isolated environments only.