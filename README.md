# c2-classifier

A command-line tool for detecting Command & Control (C2) traffic in network captures using machine learning. Extracts bidirectional flow features from raw PCAPs, classifies flows using a trained Random Forest model, and produces a ranked JSON report with per-prediction SHAP explainability.

Built as a transparent, auditable alternative to black-box network detection — every alert ships with feature attribution, every model artefact carries its full training provenance, and the validation methodology explicitly tests cross-family generalization rather than reporting only test-set metrics.

---

## Why this exists

Most ML-based intrusion detection projects on GitHub stop at "F1 = 0.99 on CIC-IDS-2017." That number is meaningless without independent validation. This project documents the full detection-engineering iteration cycle:

1. Train a Random Forest on the CIC-IDS-2017 botnet subset (v1)
2. Validate against an independent capture from malware-traffic-analysis.net (Emotet+Trickbot) — discover that v1 detects 0 of 154 flows
3. Run a controlled feature ablation to test the root-cause hypothesis (v2)
4. Confirm that single-family training causes multi-feature memorization that ablation alone cannot fix
5. Train on a multi-family corpus (CTU-13) and re-validate — v3 detects 47 of 154 flows

The result table at the bottom of this README is the actual story this project tells.

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
- ~10 GB free disk space if training on the full CTU-13 dataset

### 1. Install

```bash
git clone https://github.com/Byt3-B34r/c2-classifier.git
cd c2-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Bootstrap: train a model

> **Note.** Trained model bundles are not committed to the repository (the `models/` directory is gitignored — bundles are large binaries and tied to specific training-data versions). On a fresh clone you must train a model before `classify.py` can run. The fastest path to a working v3-equivalent classifier is the CTU-13 pipeline below.

Download the CTU-13 dataset (700 MB compressed, ~8 GB extracted):

```bash
mkdir -p data/raw && cd data/raw
curl -L -O https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2
tar -xjf CTU-13-Dataset.tar.bz2
cd ../..
```

Preprocess and train (~3 minutes total on a modern laptop):

```bash
python scripts/preprocess_ctu13.py \
    --input "data/raw/CTU-13-Dataset/*/*.binetflow" \
    --output data/processed/ctu13_all.csv \
    --benign-sample 200000 \
    -v

python scripts/train.py \
    --csv data/processed/ctu13_all.csv \
    --model models/rf_v3.joblib \
    --split time \
    --dataset-name 'CTU-13 (multi-family botnet)' \
    -v
```

### 3. Classify a PCAP

```bash
python scripts/classify.py \
    --pcap captures/sample.pcap \
    --model models/rf_v3.joblib \
    --output results.json \
    --csv results.csv \
    --top 20 \
    -v
```

The trained model lives in `models/rf_v3.joblib` and is reusable across PCAPs. Retrain only when training data or feature schema changes.

### Alternative training paths

If you want to reproduce the v1 result (CIC-IDS-2017 single-family baseline) for comparison purposes, see [Validation Methodology and Findings](#validation-methodology-and-findings) below for the full v1/v2/v3 reproduction commands.

---

## Project Structure

```
c2-classifier/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                  # PCAPs and source datasets (gitignored)
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
│   ├── preprocess_ctu13.py   # CTU-13 binetflow → project schema CSV
│   ├── train.py              # offline training pipeline
│   └── classify.py           # inference CLI
├── models/
│   └── rf_v3.joblib          # serialized trained model (gitignored)
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

> **Note on entropy and IAT-distribution features.** When training on pre-extracted CSVs (CIC-IDS-2017 or CTU-13 binetflows), several features (`payload_entropy`, `iat_skew`, `iat_kurtosis`, `dns_query_entropy`) are zero-filled because Argus/CICFlowMeter don't preserve raw payload or full IAT sequences. These features become populated only when training on raw PCAPs through this project's full pipeline. v4 work will exercise them.

---

## Training Data

| Dataset | Description | Source |
|---|---|---|
| CIC-IDS-2017 | Pre-extracted flow CSVs, Botnet ARES + benign | [UNB CIC](https://www.unb.ca/cic/datasets/ids-2017.html) |
| CTU-13 | 13-scenario botnet corpus (Neris, Rbot, Virut, Murlo, Menti, Sogou, NSIS.ay, etc.) | [Stratosphere IPS](https://www.stratosphereips.org/datasets-ctu13) |
| MTA-net (Emotet+Trickbot) | Real malware C2 PCAP, independent validation | [malware-traffic-analysis.net](https://www.malware-traffic-analysis.net/2020/02/06/index.html) |
| Synthetic lab *(planned, v5)* | Self-generated Sliver/Havoc C2 in isolated VM | Local |

Raw PCAPs are not included; `data/raw/` is gitignored. Each dataset has its own download and licensing terms.

---

## Validation Methodology and Findings

The classifier was developed iteratively, with each version designed as a hypothesis test against the previous version's failure mode. All versions were evaluated on both their own held-out test set *and* an independent capture (2020-02-06 Emotet+Trickbot from malware-traffic-analysis.net) the model had never seen.

### Test-set metrics

| Model | Training data | Precision | Recall | F1 | FPR | FNR | PR-AUC | MCC |
|---|---|---|---|---|---|---|---|---|
| **v1** | CIC-IDS-2017 (Botnet ARES) | 0.867 | 0.963 | 0.913 | 0.58% | 3.68% | 0.971 | 0.911 |
| **v2** | v1 with `dst_port`+`protocol` ablated | 0.611 | 0.984 | 0.754 | 2.45% | 1.64% | 0.898 | 0.765 |
| **v3** | CTU-13 (multi-family botnet) | 0.992 | 0.985 | 0.989 | 6.84% | 1.53% | 0.999 | 0.889 |

### Independent validation (Trickbot PCAP, 154 flows)

| Model | Flagged (≥0.5) | Highest p(c2) | Detection rate | Top feature importance |
|---|---|---|---|---|
| **v1** | 0 / 154 | 0.145 | 0% | `dst_port` (21%) |
| **v2** | 0 / 154 | 0.173 | 0% | `bwd_total_bytes` (12%) |
| **v3** | **47 / 154** | **0.812** | **31%** | `pkts_per_second` (13%) |

### Interpretation

**v1** achieved strong CIC test-set metrics (F1 = 0.913, MCC = 0.911) but failed to detect any C2 flow in the independent Trickbot capture. Feature importance was heavily concentrated on `dst_port` (21%) — suggesting port-pattern memorization specific to Botnet ARES's port range.

**v2** ablated `dst_port` and `protocol` to test the port-memorization hypothesis. Result: CIC F1 dropped 17 points (0.913 → 0.754), FPR quadrupled (0.58% → 2.45%) — but Trickbot detection did not improve. This proved the model's reliance on Botnet-ARES-specific signatures was encoded across multiple correlated features (byte distributions, packet length statistics, flow-shape ratios), not just the destination port. **Surgical feature ablation cannot remediate training-data narrowness.**

**v3** addressed the root cause by switching to CTU-13's 13-family botnet corpus (Neris, Rbot, Virut, Murlo, Menti, Sogou, NSIS.ay, and others). With training data spanning multiple malware families, the model could no longer memorize a single family's signature and was forced to learn family-agnostic flow characteristics. Result: 47 of 154 Trickbot flows correctly flagged as C2, including the documented C2 IOC IPs `203.176.135.102:8082`, `45.79.223.161:443`, and `98.239.119.52:80`. Top feature importance shifted from `dst_port` (4% in v3, down from 21% in v1) to rate-based features (`pkts_per_second`, `avg_payload_size`, `bytes_per_second`) that describe inherent C2 behavior regardless of port.

### Honest caveats

- **v3 FPR of 6.84% is too high for production SOC deployment.** The CTU-13 dataset is near-balanced (1:0.4 after subsampling), which makes the model more willing to call traffic C2. A production-tuned version would need probability threshold calibration or class-weight rebalancing.
- **v3 was trained on netflow-level features.** Argus binetflows omit packet-length distribution, IAT statistics, and payload entropy. The 31% detection rate is a baseline; v4 (training on raw CTU-13 PCAPs through the full pipeline) should improve substantially.
- **CTU-13 captures are from 2011.** Trickbot (2016+) was not in the training set. The 31% detection rate represents zero-shot generalization across nine years of malware evolution.

### v4 plan

- Process raw CTU-13 PCAPs through `parser.py` → `flow_builder.py` → `features.py` to populate the entropy and IAT-distribution features that binetflows omit
- Threshold-calibrate v3 to drop FPR below 1% while preserving recall
- Re-validate against additional MTA-net samples (Cobalt Strike, Mythic, Sliver-generated lab data)

---

## Output Format

```json
{
  "schema_version": 1,
  "source": "data/test/2020-02-06-Trickbot.pcap",
  "analyzed_at": "2026-05-10T22:51:41+00:00",
  "total_flows": 154,
  "flagged_flows": 47,
  "flag_threshold": 0.5,
  "model": {
    "trained_at": "2026-05-10T22:49:48+00:00",
    "dataset": "CTU-13 (multi-family botnet)",
    "n_train": 483524,
    "n_test": 161175,
    "class_counts": {"benign": 183967, "c2": 299557},
    "metrics": {
      "precision": 0.992, "recall": 0.985, "f1": 0.989,
      "fpr": 0.0684, "mcc": 0.889
    }
  },
  "flows": [
    {
      "flow_id": "10.20.30.101:49697-80.86.91.91:8080-6",
      "label": "c2",
      "confidence": 0.418,
      "proba_c2": 0.418,
      "src_ip": "10.20.30.101", "dst_ip": "80.86.91.91",
      "dst_port": 8080, "protocol": "TCP",
      "duration_s": 57.73, "total_packets": 1373,
      "beacon_score": 9.853, "payload_entropy": 2.717,
      "pkts_per_second": 23.79,
      "shap": {
        "pkts_per_second": 0.143,
        "avg_payload_size": 0.082,
        "bytes_per_second": 0.061,
        "flow_duration": -0.029
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

**Class imbalance.** Training defaults to `class_weight='balanced'` (sklearn) or auto-derived `scale_pos_weight` (XGBoost). Class distribution is logged at train time.

**Imbalance-honest metrics.** ROC-AUC is reported but not relied upon. PR-AUC, balanced accuracy, MCC, and FPR/FNR are the primary evaluation metrics — ROC-AUC inflates artificially on imbalanced data.

**Feature schema is append-only.** `FEATURE_NAMES` defines the model contract; new features are appended, never inserted or removed mid-list. Saved model bundles record their schema and `load()` warns on mismatch.

**Bundle versioning.** Saved models include a `bundle_version` integer. `load()` rejects bundles newer than the running code understands; older bundles load with backward-compatible defaults.

**Temporal leakage.** Time-based train/test splitting is supported via `--split time` when timestamps are available. CIC-IDS-2017 MachineLearningCVE strips timestamps; CTU-13 preserves them.

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

### `scripts/preprocess_ctu13.py`

Convert CTU-13 binetflow files into the project's feature schema.

| Flag | Purpose |
|---|---|
| `--input PATH` | Glob or single path to `.binetflow` file(s) |
| `--output PATH` | Output CSV |
| `--benign-sample N` | Subsample Normal class to N rows for class balance |
| `--include-background` | Treat Background flows as benign (not recommended) |

### `scripts/train.py`

Train a classifier from a preprocessed CSV or raw PCAP+labels.

| Flag | Purpose |
|---|---|
| `--csv PATH` | Input feature CSV (preprocessed) |
| `--pcap PATH --labels PATH` | Alternative: raw PCAP + flow_id-to-label CSV |
| `--model PATH` | Output bundle path |
| `--estimator {rf,xgboost}` | Classifier choice; default rf |
| `--split {random,time}` | Train/test split strategy |
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
- [x] `model.py` — train/save/load/predict with SHAP and imbalance-honest metrics
- [x] `report.py` — JSON, CSV, and stdout output with SHAP attribution
- [x] `preprocess_cic.py` — CIC-IDS-2017 ingestion
- [x] `preprocess_ctu13.py` — CTU-13 binetflow ingestion
- [x] `train.py` and `classify.py` CLI entrypoints
- [x] **v1** — CIC-IDS-2017 (Botnet ARES) — F1 = 0.913 test set / 0% Trickbot detection
- [x] **v2** — controlled port-feature ablation — confirmed multi-feature artifact dependence
- [x] **v3** — CTU-13 multi-family — F1 = 0.989 test set / **31% Trickbot detection**
- [ ] **v4** — raw CTU-13 PCAPs through full pipeline with entropy + IAT features
- [ ] **v5** — synthetic Sliver/Cobalt Strike lab data for modern C2 training
- [ ] `tests/test_features.py` — formal test layer for feature math
- [ ] Threshold calibration to drop v3 FPR below 1%
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