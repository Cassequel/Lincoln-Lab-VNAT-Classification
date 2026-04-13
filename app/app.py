"""
Streamlit demo — Adversarial-Robust Network Traffic Classifier
Run: streamlit run app/app.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")   # allow `from src.X import Y` from project root

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VNAT Traffic Classifier",
    page_icon="🔒",
    layout="wide",
)

# ── Load models and reference data ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    cat_model     = joblib.load("models/category_best.joblib")
    vpn_model     = joblib.load("models/vpn_best.joblib")
    features_df   = pd.read_csv("data/features.csv")
    X_ref         = features_df.drop(columns=["is_vpn", "app", "category"])
    y_ref         = features_df["category"]
    explainer     = shap.TreeExplainer(cat_model)
    return cat_model, vpn_model, X_ref, y_ref, explainer

cat_model, vpn_model, X_ref, y_ref, explainer = load_artifacts()

CATEGORY_COLORS = {
    "CHAT":          "#4C9BE8",
    "STREAMING":     "#E87C4C",
    "VOIP":          "#4CE87C",
    "FILE_TRANSFER": "#C84CE8",
    "C2":            "#E84C4C",
}

# ── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.title("Input Flow")
input_mode = st.sidebar.radio(
    "Select input method",
    ["Choose example flow", "Upload feature CSV row"],
)

if input_mode == "Choose example flow":
    category_filter = st.sidebar.selectbox(
        "Filter by category",
        ["Any"] + sorted(y_ref.unique()),
        help="Narrow the sample pool to flows of a specific traffic type. 'Any' shows all categories.",
    )
    if category_filter != "Any":
        pool = X_ref[y_ref == category_filter]
        pool_labels = y_ref[y_ref == category_filter]
    else:
        pool = X_ref
        pool_labels = y_ref

    sample_idx = st.sidebar.slider(
        "Sample index",
        0, len(pool) - 1, 0,
        help="Scroll through individual flows in the selected category pool. Each index is one network flow from the test set.",
    )
    x_input = pool.iloc[[sample_idx]]
    true_label = pool_labels.iloc[sample_idx]
    st.sidebar.caption(f"True label: **{true_label}**")

else:
    uploaded = st.sidebar.file_uploader(
        "Upload a single-row CSV with the same columns as data/features.csv",
        type="csv",
        help="The CSV must have exactly one data row and the same feature column names as data/features.csv. Label columns (is_vpn, app, category) are ignored if present.",
    )
    if uploaded is None:
        st.info("Upload a CSV row in the sidebar to get a prediction.")
        st.stop()
    x_input = pd.read_csv(uploaded)
    # Drop label columns if accidentally included
    x_input = x_input[[c for c in x_input.columns if c in X_ref.columns]]
    true_label = None

# ── Predictions ─────────────────────────────────────────────────────────────────
cat_probs  = cat_model.predict_proba(x_input)[0]
cat_pred   = cat_model.classes_[cat_probs.argmax()]
vpn_probs  = vpn_model.predict_proba(x_input)[0]
vpn_pred   = "VPN" if vpn_probs[1] > 0.5 else "Non-VPN"
vpn_conf   = vpn_probs[1] if vpn_probs[1] > 0.5 else vpn_probs[0]

# ── Main layout ─────────────────────────────────────────────────────────────────
st.title("🔒 VNAT Network Traffic Classifier")
st.caption(
    "Classifies encrypted/tunneled network flows by traffic category and VPN status "
    "using only packet metadata — no payload inspection. "
    "Built on MIT Lincoln Laboratory's VNAT dataset."
)

with st.expander("About this app / How to use it", expanded=False):
    st.markdown("""
    ### What does this do?
    Modern network traffic is almost entirely encrypted, which means you can't read the payload to figure out
    what kind of traffic it is. This app uses **machine learning on packet metadata alone** (timing, packet sizes,
    inter-arrival times, flow statistics — never the payload content) to classify a network flow into:

    - **Traffic category** — what kind of application generated it: Streaming, Chat, VoIP, File Transfer, or C2 (command-and-control / malware-like)
    - **VPN status** — whether the flow is tunneled through a VPN

    The models were trained on the [VNAT dataset](https://www.ll.mit.edu/r-d/datasets/vpnnonvpn-network-application-traffic-dataset-vnat) from MIT Lincoln Laboratory,
    which contains labeled packet captures of real application traffic.

    ---
    ### How to use it
    1. **Pick an input method** in the left sidebar:
       - *Choose example flow* — browse flows from the test dataset by category and index
       - *Upload feature CSV row* — upload your own pre-computed feature row (same columns as `data/features.csv`)
    2. **Read the prediction cards** at the top — Traffic Category and VPN Status with confidence scores.
       If you're using an example flow, a third card shows the ground-truth label so you can see if the model got it right.
    3. **Explore the charts** below:
       - *Category confidence scores* — how confident the model is across all categories
       - *Feature importance (SHAP)* — which features drove this specific prediction
       - *PCA scatter* — where this flow sits relative to all training flows
    4. **Expand "Raw feature values"** at the bottom to see the exact numbers fed into the model.

    ---
    ### What is a "flow"?
    A network flow is a sequence of packets between two endpoints sharing the same source/destination IP, port, and protocol.
    Each flow is summarized into ~30 statistical features (mean packet size, inter-arrival time variance, etc.) — no raw bytes, no payload.
    """)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    color = CATEGORY_COLORS.get(cat_pred, "#888")
    st.markdown(
        f"<div style='background:{color}22;border-left:4px solid {color};"
        f"padding:12px;border-radius:4px'>"
        f"<div style='font-size:12px;color:{color};font-weight:600'>TRAFFIC CATEGORY</div>"
        f"<div style='font-size:28px;font-weight:700'>{cat_pred}</div>"
        f"<div style='font-size:13px;color:#888'>{cat_probs.max():.1%} confidence</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col2:
    vpn_color = "#E84C4C" if vpn_pred == "VPN" else "#4C9BE8"
    st.markdown(
        f"<div style='background:{vpn_color}22;border-left:4px solid {vpn_color};"
        f"padding:12px;border-radius:4px'>"
        f"<div style='font-size:12px;color:{vpn_color};font-weight:600'>VPN STATUS</div>"
        f"<div style='font-size:28px;font-weight:700'>{vpn_pred}</div>"
        f"<div style='font-size:13px;color:#888'>{vpn_conf:.1%} confidence</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

with col3:
    if true_label:
        correct = cat_pred == true_label
        icon    = "✅" if correct else "❌"
        st.markdown(
            f"<div style='background:#88888822;border-left:4px solid #888;"
            f"padding:12px;border-radius:4px'>"
            f"<div style='font-size:12px;color:#888;font-weight:600'>TRUE LABEL</div>"
            f"<div style='font-size:28px;font-weight:700'>{true_label} {icon}</div>"
            f"<div style='font-size:13px;color:#888'>Ground truth from dataset</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ── Category probability bar chart ──────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Category confidence scores")
    prob_df = pd.DataFrame({
        "Category": cat_model.classes_,
        "Probability": cat_probs,
    }).sort_values("Probability", ascending=True)

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(prob_df["Category"].to_numpy(), prob_df["Probability"].to_numpy(),
                   color=[CATEGORY_COLORS.get(c, "#888") for c in prob_df["Category"]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.axvline(0.5, color="#888", linestyle="--", linewidth=0.8)
    for bar, prob in zip(bars, prob_df["Probability"]):
        ax.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{prob:.1%}", va="center", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── SHAP feature importance for this prediction ─────────────────────────────────
with col_b:
    st.subheader("Feature importance (SHAP) for this flow")
    shap_vals = explainer.shap_values(x_input)
    if isinstance(shap_vals, list):
        shap_vals = np.stack(shap_vals, axis=-1)  # (1, n_feats, n_classes)

    pred_class_idx = list(cat_model.classes_).index(cat_pred)
    shap_for_pred  = shap_vals[0, :, pred_class_idx]  # (n_feats,)

    shap_series = pd.Series(shap_for_pred, index=x_input.columns)
    top_pos = shap_series.nlargest(8)
    top_neg = shap_series.nsmallest(5)
    top_combined = pd.concat([top_pos, top_neg]).sort_values()

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    colors = ["#E84C4C" if v > 0 else "#4C9BE8" for v in top_combined]
    ax2.barh(top_combined.index, top_combined.values, color=colors)
    ax2.axvline(0, color="#888", linewidth=0.8)
    ax2.set_xlabel(f"SHAP value for '{cat_pred}'")
    ax2.set_title("Red = pushes toward this class", fontsize=9)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()
    st.caption(
        "**How to read this:** Each bar shows how much a feature pushed the model toward (red) or away from (blue) "
        f"the predicted class **{cat_pred}**. Longer bars = stronger influence. "
        "Features are packet-level statistics — e.g. mean inter-arrival time, payload length variance — never raw payload content."
    )

st.divider()

# ── Cluster context: where does this flow sit? ──────────────────────────────────
st.subheader("Flow in context — distribution vs. training data")

from sklearn.decomposition import PCA

@st.cache_data
def compute_pca(X):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return pca, coords

pca, ref_coords = compute_pca(X_ref)
this_coord = pca.transform(x_input)

fig3, ax3 = plt.subplots(figsize=(8, 4))
for cat in y_ref.unique():
    mask = y_ref == cat
    ax3.scatter(ref_coords[mask, 0], ref_coords[mask, 1],
                c=CATEGORY_COLORS.get(cat, "#888"), label=cat,
                alpha=0.3, s=8, linewidths=0)

ax3.scatter(this_coord[0, 0], this_coord[0, 1],
            c="white", edgecolors="black", s=150, zorder=5,
            marker="*", label="This flow")
ax3.set_xlabel("PCA component 1")
ax3.set_ylabel("PCA component 2")
ax3.set_title("PCA of engineered features (training data)")
ax3.legend(markerscale=2, fontsize=8, loc="best")
fig3.tight_layout()
st.pyplot(fig3)
plt.close()
st.caption(
    "**How to read this:** Each dot is a flow from the training set, colored by its true category. "
    "The ★ is the flow you're inspecting. PCA compresses ~30 features into 2 dimensions so you can see "
    "whether this flow sits cleanly inside its predicted cluster or near a boundary (which may explain lower confidence)."
)

# ── Raw feature values ──────────────────────────────────────────────────────────
with st.expander("Raw feature values for this flow"):
    st.dataframe(x_input.T.rename(columns={x_input.index[0]: "value"}).style.format("{:.4f}"))
