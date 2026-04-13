# Adversarial-Robust Network Traffic Classification

Using MIT Lincoln Laboratory's [VNAT dataset](https://www.ll.mit.edu/r-d/datasets/vnat-dataset) to build and stress-test an encrypted traffic classifier — directly demonstrating work relevant to Lincoln Lab's **AI Technology and Systems Group (Group 5-52)**.

---

## Problem

When network traffic is encrypted or tunneled through a VPN, traditional deep packet inspection fails. A network defender sees opaque UDP/TCP streams with no payload visibility — making it impossible to distinguish legitimate streaming from command-and-control (C2) infrastructure using conventional signatures.

Lincoln Lab created the VNAT dataset to study exactly this problem. This project replicates and extends their approach:

1. Classify encrypted traffic by **traffic category** (Chat, Streaming, VoIP, File Transfer, C2) using only packet metadata — no payload inspection
2. Classify **VPN status** (tunneled vs. cleartext) from the same metadata
3. Attack the classifier with adversarial perturbations and measure how much manipulation an adversary needs to evade detection
4. Harden the model via adversarial training and quantify the improvement

---

## Dataset

**Source:** MIT Lincoln Laboratory VNAT (VPN/Non-VPN Network Application Traffic), Release 1  
**Registration required:** [https://www.ll.mit.edu/r-d/datasets/vnat-dataset](https://www.ll.mit.edu/r-d/datasets/vnat-dataset)

| File | Rows | Description |
|------|------|-------------|
| `VNAT_Dataframe_release_1.h5` | 33,711 | Raw flows: connection tuples, timestamps, packet sizes, directions |
| `VNAT_Feature_Dataframe_release_1.h5` | 15,093 | Pre-computed feature matrix (no VPN labels — unusable for joint classification) |

**Labels parsed from filenames** (e.g. `vpn_skype-chat_capture9.pcap`):

| Label | Type | Classes |
|-------|------|---------|
| `is_vpn` | Binary | VPN, Non-VPN |
| `category` | 5-class | CHAT, STREAMING, VOIP, FILE\_TRANSFER, C2 |
| `app` | 10-class | skype-chat, youtube, sftp, ssh, rdp, … |

---

## Methodology

```
Raw flows (33k)
    │
    ├─ Label extraction (filename parsing)
    ├─ Data quality filtering (single-packet flows dropped)
    │
    ▼
Feature engineering (src/feature_engineering.py)
    ├─ Timing:      IAT min/max/mean/std per direction; active/idle periods; burst detection
    ├─ Volume:      log bytes/packets per direction; bytes/sec; directional ratio
    ├─ Distribution: Shannon entropy of packet size histograms
    └─ Wavelet:     Haar DWT on packet size sequence — log mean/std of detail
                    coefficients at 12 decomposition levels (replicates Lincoln Lab)
    │
    ▼
Baseline classification (02_modeling.ipynb)
    ├─ Task A: 5-class traffic category (LR / Random Forest / LightGBM)
    └─ Task B: Binary VPN status
    │
    ▼
Adversarial analysis (03_adversarial.ipynb)
    ├─ Step 3.1: Perturbation sweep — misclassification rate vs. budget
    ├─ Step 3.2: Evasion attack — C2 → CHAT via greedy coordinate descent
    ├─ Step 3.3: Adversarial training — augment with adversarial C2, retrain
    └─ Step 3.4: Operational analysis — realistic vs. theoretical attack vectors
```

---

## Results

### Task A — Traffic Category Classification (5-class, 20% hold-out test set)

| Model | Macro F1 | CHAT F1 | VOIP F1 | FILE\_TRANSFER F1 | STREAMING F1 | C2 F1 |
|-------|----------|---------|---------|-------------------|--------------|-------|
| Logistic Regression | 0.698 | 0.948 | 0.560 | 0.963 | 0.955 | 0.064 |
| Random Forest | 0.858 | 0.996 | 1.000 | 0.998 | 1.000 | 0.296 |
| **LightGBM** | **0.865** | **0.996** | **1.000** | **0.997** | **1.000** | **0.333** |

C2 is the hardest class (covert traffic deliberately mimics legitimate flows); wavelet features provide the strongest signal for distinguishing it.

### Task B — VPN Status Classification (binary, 20% hold-out test set)

| Model | Macro F1 | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 0.966 | 0.9999 |
| Random Forest | 0.996 | 1.0000 |
| **LightGBM** | **1.000** | **1.0000** |

VPN classification is near-perfect: tunneling leaves a distinctive packet-size and timing signature even without payload access.

### Adversarial Robustness (greedy coordinate descent, 20 highest-variance features)

| Metric | Original LightGBM | Hardened (adv-trained) |
|--------|:-----------------:|:----------------------:|
| C2 → CHAT evasion success rate | 14% (1/7) | 29% (2/7) |
| Mean budget to flip any class (× feat std) | 4.56 | 4.03 |
| C2 per-class robustness score | 3.64 | 3.64 |
| Misclassification at budget = 1σ | 6.3% | 14.7% |
| Misclassification at budget = 5σ | 15.0% | 33.3% |

The adversarial-training pass (augmenting with perturbed C2 flows and retraining) did not improve robustness under random perturbation — a common finding when training perturbations are targeted but evaluation perturbations are stochastic. The C2 evasion budget required (≈3.6σ mean) confirms that an adversary must make large, operationally visible changes to evade the detector.

---

## Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the VNAT dataset

Register at [https://www.ll.mit.edu/r-d/datasets/vnat-dataset](https://www.ll.mit.edu/r-d/datasets/vnat-dataset) and place the two `.h5` files in the project root (or `data/`). See `data/download_data.sh` for a curl template.

### 3. Run the notebooks in order

```
01_eda.ipynb          → parses labels, engineers features, saves data/features.csv
02_modeling.ipynb     → trains classifiers, saves models/
03_adversarial.ipynb  → evasion attacks, adversarial training, operational analysis
```

### 4. Run the interactive demo

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`. Select any flow from the dataset, see real-time predictions, SHAP feature attributions, and a PCA context plot.

**Streamlit Cloud deployment:** Fork the repo, go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set `app/app.py` as the entry point. Models and features are committed (`models/`, `data/features.csv`) — no extra setup needed.

### 5. Or run the training script directly

```bash
# Train best category classifier and save to models/
python src/train.py --task category --model lgbm

# Train VPN classifier
python src/train.py --task vpn --model lgbm

# Compare all three model families
python src/train.py --task category --model all
```

---

## Repository Structure

```
├── 01_eda.ipynb                  # EDA + feature engineering
├── 02_modeling.ipynb             # Baseline classification
├── 03_adversarial.ipynb          # Adversarial robustness analysis
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml               # Streamlit Cloud configuration
├── app/
│   └── app.py                    # Streamlit interactive demo
├── data/
│   ├── download_data.sh          # Dataset download instructions
│   └── features.csv              # Engineered feature matrix (23k flows, 116 features)
├── models/                       # Saved model artifacts (LR, RF, LightGBM × 2 tasks + hardened)
└── src/
    ├── feature_engineering.py    # extract_features() pipeline
    ├── train.py                  # CLI training script
    └── adversarial.py            # Perturbation and evasion attack library
```

---

## Key Design Decisions

**Why engineer features from scratch instead of using the pre-computed matrix?**  
Lincoln Lab's pre-computed HDF5 drops VPN labels and has a different row count (15k vs 33k), making joint VPN + category classification impossible from that file alone. Engineering features from the raw dataframe recovers all labels and demonstrates full pipeline understanding.

**Why use wavelet features?**  
Lincoln Lab's own feature matrix includes Haar DWT coefficients at 12 decomposition levels. Replicating this shows direct familiarity with their methodology. Wavelet features capture multi-scale periodicity in packet size sequences — VOIP has fine-grained regularity; file transfers have coarse bursts — in a fixed-length vector regardless of flow length.

**Why constrain adversarial perturbations?**  
An attacker can pad packets (increase sizes) or inject delays (increase inter-arrival times), but cannot shrink packets already transmitted or reverse time. Ignoring these physical constraints produces evasion results that look impressive but are operationally meaningless. All perturbations in this project enforce one-directional constraints on timing and size features.
