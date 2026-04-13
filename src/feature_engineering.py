"""
Feature extraction pipeline for VNAT raw flow data.

Usage:
    from src.feature_engineering import extract_features, app_to_category
    feat_matrix = df.progress_apply(extract_features, axis=1)
"""

import numpy as np
import pywt

# Cap signal length fed to wavelet -- flows with millions of packets (e.g.
# scp_long) would otherwise take minutes per row.
MAX_PKTS     = 2048
WAVELET      = "haar"
WAV_LEVELS   = 12
IDLE_THRESH  = 1.0   # seconds -- gap >= this is classified as 'idle'
BURST_THRESH = 0.1   # seconds -- gap >= this breaks a burst

app_to_category = {
    "netflix":    "STREAMING",
    "youtube":    "STREAMING",
    "vimeo":      "STREAMING",
    "voip":       "VOIP",
    "skype-chat": "CHAT",
    "sftp":       "FILE_TRANSFER",
    "scp":        "FILE_TRANSFER",
    "scp_long":   "FILE_TRANSFER",
    "rsync":      "FILE_TRANSFER",
    "ssh":        "FILE_TRANSFER",
    "rdp":        "C2",
}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _iat_stats(times: np.ndarray, tag: str, feats: dict) -> np.ndarray:
    """Fill inter-arrival time statistics for a direction tag."""
    if len(times) > 1:
        iats = np.diff(times)
        feats[f"{tag}_iat_min"]  = float(iats.min())
        feats[f"{tag}_iat_max"]  = float(iats.max())
        feats[f"{tag}_iat_mean"] = float(iats.mean())
        feats[f"{tag}_iat_std"]  = float(iats.std())
        return iats
    for s in ("min", "max", "mean", "std"):
        feats[f"{tag}_iat_{s}"] = 0.0
    return np.array([0.0])


def _size_stats(sizes: np.ndarray, tag: str, feats: dict) -> None:
    """Fill packet size statistics for a direction tag."""
    if len(sizes) > 0:
        feats[f"{tag}_size_min"]    = float(sizes.min())
        feats[f"{tag}_size_max"]    = float(sizes.max())
        feats[f"{tag}_size_mean"]   = float(sizes.mean())
        feats[f"{tag}_size_std"]    = float(sizes.std())
        feats[f"{tag}_size_median"] = float(np.median(sizes))
        feats[f"{tag}_size_p25"]    = float(np.percentile(sizes, 25))
        feats[f"{tag}_size_p75"]    = float(np.percentile(sizes, 75))
    else:
        for s in ("min", "max", "mean", "std", "median", "p25", "p75"):
            feats[f"{tag}_size_{s}"] = 0.0


def _entropy(sizes: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy (bits) of binned packet size distribution."""
    if len(sizes) == 0:
        return 0.0
    counts, _ = np.histogram(sizes, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _wavelet_features(sizes: np.ndarray, tag: str, feats: dict,
                      levels: int = WAV_LEVELS) -> None:
    """
    Log mean/std of DWT detail coefficients and relative energy at each level.
    Replicates Lincoln Lab's wavelet feature naming convention.
    """
    sig = sizes[:MAX_PKTS].astype(float)
    if len(sig) < 2:
        sig = np.pad(sig, (0, 2 - len(sig)))
    coeffs = pywt.wavedec(sig, WAVELET, level=levels, mode="periodization")
    # coeffs = [cA_N, cD_N, cD_{N-1}, ..., cD_1]
    detail = coeffs[1:]
    total_energy = sum(np.sum(c ** 2) for c in coeffs)
    for i, cd in enumerate(detail, start=1):
        feats[f"{tag}_log_mean_detail_coeffs_{i}"]    = float(np.log1p(np.abs(cd).mean()))
        feats[f"{tag}_log_std_dev_detail_coeffs_{i}"] = float(np.log1p(cd.std()))
        level_energy = float(np.sum(cd ** 2))
        feats[f"{tag}_rel_energy_{i}"] = level_energy / total_energy if total_energy > 0 else 0.0


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_features(row) -> dict:
    """
    Extract all features from a single raw-flow row.

    Parameters
    ----------
    row : pandas Series with fields: sizes, timestamps, directions

    Returns
    -------
    dict mapping feature name -> float value
    """
    sizes = np.array(row["sizes"],      dtype=float)
    times = np.array(row["timestamps"], dtype=float)
    dirs  = np.array(row["directions"], dtype=int)   # 1=out, 0=in

    out_mask  = dirs == 1
    in_mask   = dirs == 0
    out_sizes = sizes[out_mask]
    in_sizes  = sizes[in_mask]
    out_times = times[out_mask]
    in_times  = times[in_mask]
    feats: dict = {}

    # Volume
    total_bytes = float(sizes.sum())
    duration    = float(times[-1] - times[0])
    feats["log_out_pkt_count"] = float(np.log1p(out_mask.sum()))
    feats["log_in_pkt_count"]  = float(np.log1p(in_mask.sum()))
    feats["log_out_bytes"]     = float(np.log1p(out_sizes.sum()))
    feats["log_in_bytes"]      = float(np.log1p(in_sizes.sum()))
    feats["dir_ratio"]         = float(out_sizes.sum() / total_bytes) if total_bytes > 0 else 0.5
    feats["bytes_per_sec"]     = total_bytes / duration if duration > 0 else 0.0
    _size_stats(out_sizes, "out", feats)
    _size_stats(in_sizes,  "in",  feats)

    # Timing
    flow_iats = _iat_stats(times,     "flow", feats)
    _iat_stats(out_times, "out",  feats)
    _iat_stats(in_times,  "in",   feats)

    # Active / idle periods
    idle_gaps   = flow_iats[flow_iats >= IDLE_THRESH]
    active_gaps = flow_iats[flow_iats <  IDLE_THRESH]
    feats["idle_time"]   = float(idle_gaps.sum())
    feats["active_time"] = float(active_gaps.sum())
    feats["n_idle"]      = int(len(idle_gaps))

    # Burst detection
    breaks       = np.where(flow_iats >= BURST_THRESH)[0]
    burst_pkts   = np.split(sizes, breaks + 1)
    burst_starts = np.concatenate([[0], breaks + 1])
    burst_ends   = np.concatenate([breaks + 1, [len(times)]])
    burst_durs   = [float(times[e - 1] - times[s])
                    for s, e in zip(burst_starts, burst_ends) if e > s]
    feats["n_bursts"]        = int(len(breaks) + 1)
    feats["mean_burst_pkts"] = float(np.mean([len(b) for b in burst_pkts]))
    feats["mean_burst_dur"]  = float(np.mean(burst_durs)) if burst_durs else 0.0

    # Entropy
    feats["out_size_entropy"]  = _entropy(out_sizes)
    feats["in_size_entropy"]   = _entropy(in_sizes)
    feats["flow_size_entropy"] = _entropy(sizes)

    # Wavelet
    if len(out_sizes) >= 2:
        _wavelet_features(out_sizes, "out", feats)
    if len(in_sizes) >= 2:
        _wavelet_features(in_sizes, "in", feats)

    return feats
