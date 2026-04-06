# Adversarial-Robust Network Traffic Classification

## Using MIT Lincoln Laboratory's VNAT Dataset

---

## Objective

Build a machine learning pipeline that classifies encrypted and unencrypted network traffic by **traffic category** (Chat, Streaming, VoIP, File Transfer, C2) and **VPN status** (VPN vs. Non-VPN) using only packet metadata — no payload inspection. Then stress-test the models with adversarial techniques to evaluate robustness under evasion attacks.

The project uses MIT Lincoln Laboratory's own VNAT (VPN/Non-VPN Network Application Traffic) dataset to directly demonstrate competence and interest for the **AI Technology and Systems Group (Group 5-52) Summer Research Internship**.

---

## Why This Project

Lincoln Lab created the VNAT dataset to solve a specific operational problem: when traffic is encrypted or tunneled through a VPN, traditional deep packet inspection can't identify what application generated it. A network defender sees a stream of opaque UDP packets and has no way to distinguish someone chatting on Skype from an adversary running command-and-control infrastructure.

The goal is ML-based traffic classification using only observable metadata — packet sizes, timing patterns, byte distributions, and flow statistics. This maps directly to the group's mission areas: **cyber AI/ML**, **adversarial AI**, and **transitioning ML technologies to operational environments**.

---

## What We're Working With

### Two Data Sources

**`VNAT_Dataframe_release_1.h5`** — Raw flow-level data (33,711 rows × 5 columns)
- `connection`: Tuple of (src_ip, src_port, dst_ip, dst_port, protocol)
- `timestamps`: List of packet arrival times per flow
- `sizes`: List of packet sizes per flow
- `directions`: List of packet direction indicators (inbound/outbound)
- `file names`: Source pcap filename encoding VPN status, application, and capture session

**`VNAT_Feature_Dataframe_release_1.h5`** — Pre-computed feature matrix (15,093 rows × 130 columns)
- 129 engineered numeric features (timing, volume, wavelet energy, entropy, detail coefficients)
- 1 label column with 5 traffic categories
- **Does NOT include VPN/non-VPN labels or specific application names**

### The Problem

The two dataframes have different row counts (33,711 vs 15,093) and no shared key to join them. Lincoln Lab aggregated or filtered flows during feature engineering, dropping the VPN indicator in the process. To get VPN labels alongside model-ready features, we need to engineer features ourselves from the raw dataframe.

### Three Classification Targets (Parsed from Filenames)

| Target | Type | Classes | Example |
|--------|------|---------|---------|
| **VPN Status** | Binary | VPN, Non-VPN | `vpn_skype-chat_capture9.pcap` → VPN |
| **Traffic Category** | Multi-class (5) | Chat, Streaming, VoIP, File Transfer, C2 | `vpn_skype-chat_capture9.pcap` → Chat |
| **Application** | Multi-class (10) | Skype, YouTube, SFTP, etc. | `vpn_skype-chat_capture9.pcap` → Skype |

---

## Step-by-Step Plan

### Phase 1: Data Preparation & Label Extraction

**Goal:** Parse raw dataframe filenames to extract all three label levels, then engineer features from the raw packet data.

**Step 1.1 — Load and Parse Labels**
- Load `VNAT_Dataframe_release_1.h5` (raw dataframe)
- Parse the `file names` column to extract:
  - `is_vpn`: binary flag from filename prefix (`vpn_` vs `nonvpn_`)
  - `app`: specific application name from the middle of the filename
  - `category`: map each app to one of the 5 traffic categories (Chat, Streaming, VoIP, File Transfer, C2)
- Validate label distributions and check for any ambiguous or malformed filenames

**Step 1.2 — Exploratory Data Analysis on Raw Flows**
- Examine the `sizes`, `timestamps`, and `directions` lists per flow
- Compute basic stats: flow length distributions, packet count distributions, duration distributions
- Visualize differences between VPN and non-VPN flows
- Visualize differences across the 5 traffic categories
- Identify any data quality issues (empty flows, single-packet flows, outliers)

**Step 1.3 — Feature Engineering**
- Extract features from each flow's raw packet data, aiming to replicate and extend what Lincoln Lab computed:

  **Timing features:**
  - Inter-arrival time statistics (min, max, mean, std) for outbound, inbound, and combined flow
  - Active/idle period statistics
  - Burst detection metrics (number of bursts, mean burst duration, mean burst packet count)

  **Volume features:**
  - Total bytes and packets (log-transformed), both directions
  - Bytes per second
  - Packet size statistics (min, max, mean, std, median, percentiles) per direction
  - Directional ratio (outbound bytes / total bytes)

  **Distribution features:**
  - Packet size entropy (Shannon entropy of binned size distribution) per direction
  - Byte-level entropy if accessible

  **Wavelet features (advanced, optional):**
  - Relative energy at multiple decomposition levels
  - Shannon entropy of wavelet coefficients
  - Log mean/std of detail coefficients
  - (This replicates Lincoln Lab's approach — impressive if included but not required for a strong project)

- Output: a clean dataframe with engineered features, all three label columns, and no missing values

**Deliverables:**
- `01_eda.ipynb` — Exploratory analysis with visualizations
- `src/feature_engineering.py` — Reusable feature extraction pipeline
- `data/features.csv` — Engineered feature matrix with labels

---

### Phase 2: Baseline Classification

**Goal:** Train and evaluate classifiers on the traffic category and VPN status prediction tasks.

**Step 2.1 — Data Splitting**
- Stratified train/validation/test split (70/15/15) preserving class balance
- Stratify on traffic category since it has the most imbalance (Chat ~70%, VoIP ~1.6%)

**Step 2.2 — Address Class Imbalance**
- Evaluate approaches: class weights, SMOTE oversampling, or combination
- Select method based on validation performance
- Document the imbalance ratios and chosen strategy

**Step 2.3 — Train Baseline Models**
- **Task A: Traffic Category Classification (5-class)**
  - Logistic Regression (baseline reference)
  - Random Forest
  - LightGBM or XGBoost
  - Hyperparameter tuning via cross-validation (GridSearch or Optuna)

- **Task B: VPN Status Classification (binary)**
  - Same model suite
  - Potentially simpler — VPN tunneling alters metadata patterns uniformly

- **Task C (Optional): Multi-task or Hierarchical**
  - Predict VPN status first, then traffic category conditioned on VPN status
  - Compare against flat multi-class approach

**Step 2.4 — Evaluate**
- Metrics: precision, recall, F1 (macro and per-class), confusion matrix, ROC-AUC
- Focus on per-class F1 since accuracy is misleading with 70% Chat dominance
- SHAP or feature importance analysis to identify which features drive predictions
- Compare feature importance between VPN and non-VPN subsets

**Deliverables:**
- `02_modeling.ipynb` — Model training, comparison tables, confusion matrices
- `src/train.py` — Training script with configurable model selection
- `models/` — Saved model artifacts (joblib/pickle)

---

### Phase 3: Adversarial Robustness Analysis

**Goal:** Test whether the classifier can be evaded by an adversary manipulating traffic patterns, then harden it.

This phase is the differentiator for the Lincoln Lab application. The AI Technology and Systems group explicitly works on adversarial AI — building ML that holds up under attack.

**Step 3.1 — Feature-Space Perturbation Analysis**
- Take correctly classified test samples
- Apply incremental perturbations to key features (packet sizes, timing statistics)
- Measure what percentage of perturbation budget causes misclassification
- Compute a per-class "robustness score"
- Identify which features are most sensitive to perturbation (attack surface analysis)

**Step 3.2 — Evasion Attack Simulation**
- Implement a simple evasion strategy: given a C2 flow, what minimal feature modifications would make it classify as Chat?
- Use gradient-based approach (for differentiable models) or iterative perturbation (for tree models)
- Constrain perturbations to be realistic — an attacker can pad packets or add delays, but can't shrink packets already sent
- Report evasion success rate and required perturbation magnitude

**Step 3.3 — Adversarial Training (Hardening)**
- Generate adversarial examples from Step 3.2
- Augment training data with adversarial samples
- Retrain the best model on the augmented dataset
- Compare robustness before and after:
  - Clean accuracy (should stay high)
  - Adversarial accuracy (should improve)
  - Per-class robustness scores

**Step 3.4 — Operational Analysis**
- Frame results in operational terms: "An adversary would need to modify X% of packet timing features to evade detection of C2 traffic as Chat"
- Discuss which attack vectors are realistic vs. theoretical
- Recommend detection strategies for the most viable evasion approaches

**Deliverables:**
- `03_adversarial.ipynb` — Attack methodology, robustness metrics, before/after comparison
- `src/adversarial.py` — Perturbation and evasion attack implementations

---

### Phase 4: Documentation & Packaging

**Goal:** Polish everything into a portfolio-ready GitHub repository.

**Step 4.1 — Repository Structure**
```
adversarial-traffic-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── download_data.sh        # Script to fetch VNAT from Lincoln Lab
│   └── features.csv            # Engineered features (gitignored if large)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_adversarial.ipynb
├── src/
│   ├── feature_engineering.py
│   ├── train.py
│   └── adversarial.py
├── models/                     # Saved model artifacts
└── app/                        # Optional Streamlit demo
    └── app.py
```

**Step 4.2 — README**
- Clear problem statement referencing Lincoln Lab's operational mission
- Methodology overview with architecture diagram
- Key results (accuracy, robustness metrics, feature importance findings)
- How to reproduce (install, download data, run pipeline)
- Link to VNAT dataset on Lincoln Lab's data portal

**Step 4.3 — Optional Streamlit Demo**
- Upload or select a network flow
- Show predicted traffic category + VPN status with confidence scores
- Display feature importance for the specific prediction
- Visualize where the flow falls relative to training data clusters

**Step 4.4 — Deploy Web App**
- Deploy the Streamlit demo to [Streamlit Community Cloud](https://streamlit.io/cloud) (free, GitHub-connected)
  - Push the `app/` directory to the GitHub repo
  - Connect the repo to Streamlit Cloud and set the entry point to `app/app.py`
  - Add any required secrets (e.g. model paths) via Streamlit Cloud's secrets manager
- Ensure the app loads a pre-trained model artifact from `models/` rather than retraining on launch
- Add the live app URL to the README and resume line
- Alternatively, containerize with Docker and deploy to a cloud provider (Render, Fly.io, or HuggingFace Spaces) if more control over the runtime environment is needed

---

## Timeline

| Phase | Days | Key Output |
|-------|------|------------|
| 1 — Data Prep & Feature Engineering | 1–5 | Parsed labels, engineered features, EDA notebook |
| 2 — Baseline Classification | 5–9 | Tuned models, SHAP analysis, per-class metrics |
| 3 — Adversarial Robustness | 9–12 | Evasion attacks, adversarial training, hardened model |
| 4 — Documentation & Packaging | 12–14 | GitHub repo, README, optional Streamlit app |

---

## Resume Line (Draft)

> Built adversarial-robust network traffic classifier using MIT Lincoln Lab's VNAT dataset; engineered features from raw packet captures to predict traffic category (5-class, F1: X%) and VPN status (binary, F1: X%); implemented evasion attacks and adversarial training, improving model robustness by X% while maintaining baseline performance on clean data.

---

## Key Technical Decisions to Document

1. **Why engineer features from raw data instead of using the pre-computed feature matrix** — The pre-computed HDF5 drops VPN labels and has a different row count than the raw data, making it impossible to recover VPN status. Engineering features from scratch also demonstrates deeper understanding of the pipeline.

2. **Why traffic category is the primary target** — VPN detection is too easy (binary, distinctive patterns). Application identification is too granular for operational relevance. Traffic category is the operationally meaningful middle ground — a SOC analyst needs to know if encrypted traffic is C2 vs. streaming, not Skype vs. Hangouts.

3. **Why adversarial robustness matters** — Lincoln Lab's group explicitly works on adversarial AI for cyber. A classifier that works on clean data but fails under evasion is operationally useless. Testing and hardening against adversarial manipulation is the differentiator.

4. **Why certain perturbations are constrained** — An attacker can pad packets (increase sizes) or add delays (increase inter-arrival times) but cannot shrink already-sent packets or reverse time. Adversarial analysis must respect these physical constraints to be credible.
