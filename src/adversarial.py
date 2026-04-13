"""
Perturbation and evasion attack implementations for VNAT traffic classification.

Usage:
    from src.adversarial import is_constrained, evade_sample, compute_robustness, perturbation_sweep
"""

import numpy as np
import pandas as pd

# Constrained features: adversary can only INCREASE these
# (packets already sent cannot be shrunk; delays cannot be un-added)
_CONSTRAINED_PREFIXES = (
    "out_size", "in_size", "out_iat", "in_iat", "flow_iat",
    "bytes_per_sec", "log_out", "log_in",
)


def is_constrained(feat_name: str) -> bool:
    """Return True if this feature can only increase under a realistic attack."""
    return any(feat_name.startswith(p) for p in _CONSTRAINED_PREFIXES)


def evade_sample(x: pd.Series, model, target_class: str,
                 top_features: list, feat_std: pd.Series,
                 step_size: float = 0.05, max_steps: int = 200) -> dict:
    """
    Greedily perturb x until it classifies as target_class.

    At each step, tries +/- step_size * feat_std on every feature in
    top_features and applies the move that most increases target_class
    probability. Constrained features only allow positive deltas.

    Parameters
    ----------
    x            : single flow feature vector (pd.Series)
    model        : fitted classifier with predict_proba()
    target_class : class label to induce (e.g. 'CHAT')
    top_features : feature names to perturb
    feat_std     : per-feature standard deviations
    step_size    : perturbation step as fraction of feat_std
    max_steps    : iteration cap

    Returns
    -------
    dict with keys: success (bool), steps (int), l2_norm (float), x_adv (pd.Series)
    """
    classes    = list(model.classes_)
    target_idx = classes.index(target_class)

    x_adv       = x.copy().astype(float)
    total_delta = np.zeros(len(x_adv))

    for step in range(max_steps):
        prob = model.predict_proba(x_adv.values.reshape(1, -1))[0]
        if classes[prob.argmax()] == target_class:
            return {"success": True, "steps": step,
                    "l2_norm": float(np.linalg.norm(total_delta)),
                    "x_adv": x_adv}

        best_gain, best_feat, best_d = -np.inf, None, 0.0
        for feat in top_features:
            delta = step_size * feat_std[feat]
            directions = [delta] if is_constrained(feat) else [delta, -delta]
            for d in directions:
                x_try = x_adv.copy()
                x_try[feat] += d
                p = model.predict_proba(x_try.values.reshape(1, -1))[0]
                if p[target_idx] > best_gain:
                    best_gain, best_feat, best_d = p[target_idx], feat, d

        if best_feat is None:
            break
        feat_idx = list(x_adv.index).index(best_feat)
        x_adv[best_feat] += best_d
        total_delta[feat_idx] += best_d

    return {"success": False, "steps": max_steps,
            "l2_norm": float(np.linalg.norm(total_delta)),
            "x_adv": x_adv}


def perturbation_sweep(model, X: pd.DataFrame, y: pd.Series,
                       top_features: list, feat_std: pd.Series,
                       budgets: np.ndarray, rng=None) -> np.ndarray:
    """
    For each budget k, apply Gaussian noise (std = k * feat_std) to top_features
    and record the smallest k that causes each sample to misclassify.

    Returns flip_budget array of shape (n_samples,); np.inf = never flipped.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    flip_budget = np.full(len(X), np.inf)

    for k in budgets:
        if k == 0:
            continue
        X_p = X.copy()
        for feat in top_features:
            delta = rng.normal(0, k * feat_std[feat], size=len(X))
            if is_constrained(feat):
                delta = np.clip(delta, 0, None)
            X_p[feat] = X_p[feat] + delta

        flipped   = model.predict(X_p) != y.values
        flip_budget = np.where((flip_budget == np.inf) & flipped, k, flip_budget)

    return flip_budget


def compute_robustness(model, X: pd.DataFrame, y: pd.Series,
                       top_features: list, feat_std: pd.Series,
                       budgets: np.ndarray, rng=None) -> pd.Series:
    """
    Per-class robustness score: mean perturbation budget needed to flip samples.
    Higher = harder to evade.

    Never-flipped samples are assigned the maximum budget (conservative).
    """
    flip_budget = perturbation_sweep(model, X, y, top_features, feat_std, budgets, rng)
    max_budget  = budgets[-1]

    rob = {}
    for cls in y.unique():
        mask = y == cls
        b    = np.where(flip_budget[mask] == np.inf, max_budget, flip_budget[mask])
        rob[cls] = float(b.mean())
    return pd.Series(rob)


def feature_sensitivity(model, X: pd.DataFrame, y: pd.Series,
                        top_features: list, feat_std: pd.Series,
                        budgets: np.ndarray,
                        flip_threshold: float = 0.10,
                        rng=None) -> pd.Series:
    """
    For each feature, find the smallest budget that causes flip_threshold fraction
    of samples to misclassify when ONLY that feature is perturbed.

    Returns pd.Series mapping feature -> budget (np.inf if never reached threshold).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    sensitivity = {}
    for feat in top_features:
        for k in budgets:
            if k == 0:
                continue
            X_p = X.copy()
            delta = rng.normal(0, k * feat_std[feat], size=len(X))
            if is_constrained(feat):
                delta = np.clip(delta, 0, None)
            X_p[feat] = X_p[feat] + delta
            rate = (model.predict(X_p) != y.values).mean()
            if rate >= flip_threshold:
                sensitivity[feat] = float(k)
                break
        else:
            sensitivity[feat] = np.inf

    return pd.Series(sensitivity)
