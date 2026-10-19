"""SynthetixMed AI Governance - schema-agnostic drift helpers (CSV/XPT inputs, any condition)."""
from __future__ import annotations

import hashlib
import io
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_HINTS = ("target", "label", "outcome", "diabet", "onset", "diagnos", "class", "death",
                "mortality", "readmi", "sepsis", "stroke", "cancer", "disease")
SCENARIOS = {"Stable cohort": 0.0, "Moderate drift": 1.0, "Severe drift": 2.0}


# ------------------------------------------------------------ loading ----
def read_csv(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode CSV file")


def read_dataset(raw: bytes, filename: str) -> pd.DataFrame:
    """Read an uploaded CSV or SAS Transport XPT into a normalized DataFrame.

    XPT decoding happens in memory; no conversion or saved CSV is required.
    Both formats pass through the same missing-code cleaning in app.py.
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension == "csv":
        return read_csv(raw)
    if extension == "xpt":
        try:
            df = pd.read_sas(io.BytesIO(raw), format="xport", encoding="utf-8")
        except UnicodeDecodeError:
            # Some historical SAS files use legacy single-byte text encodings.
            df = pd.read_sas(io.BytesIO(raw), format="xport", encoding="latin-1")
        df.columns = [str(c).strip() for c in df.columns]
        return df
    raise ValueError("Unsupported format. Upload a .csv or extracted .xpt file (not a ZIP archive).")


# Sentinel codes are dataset-specific. Never replace 7/9 globally: they can be
# legitimate ages, scores, or measurements. The UI exposes additional overrides.
COMMON_TEXT_MISSING = {"", "na", "n/a", "null", "none", "nan", "unknown", "missing", "refused", "don't know", "dont know", "not asked", "not applicable"}
NEGATIVE_SENTINELS = {-1, -7, -8, -9, -77, -88, -99, -777, -888, -999, -9999}
EXTREME_SENTINELS = {7777, 8888, 9999, 77777, 88888, 99999, 777777, 888888, 999999}
SURVEY_FLAG_PATTERN = re.compile(r"(?:^|_)(?:smok|diabet|hypertens|highbp|highchol|physact|exercise|stroke|heart|asthma|sex|gender|pregnan|insured|healthcare|alcohol|tobacco|yesno|flag|indicator)", re.I)


def clean_missing_codes(df: pd.DataFrame, extra_codes: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Normalize conservative missing markers; return cleaned copy and per-column replacement counts.

    Values 7/9 are treated as missing only for survey-like flag columns whose
    observed nonmissing codes are 0/1/2/7/9. For CDC BRFSS, use the official
    codebook and supply explicit per-column overrides for other encodings.
    No generic upper-limit clipping is performed on clinical measurements.
    """
    cleaned = df.copy()
    changes = {}
    for col in cleaned:
        series = cleaned[col]
        before = series.isna().sum()
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            txt = series.astype('string').str.strip()
            series = series.mask(txt.str.lower().isin(COMMON_TEXT_MISSING).fillna(False))
        numeric = pd.to_numeric(series, errors="coerce")
        # Only apply numeric sentinels when the original column is numeric or
        # all of its nonmissing string values are numeric.
        numeric_like = pd.api.types.is_numeric_dtype(series) or (numeric.notna().sum() == series.notna().sum())
        if numeric_like:
            codes = NEGATIVE_SENTINELS | EXTREME_SENTINELS
            valid = set(numeric.dropna().unique())
            if SURVEY_FLAG_PATTERN.search(str(col)) and valid.issubset({0, 1, 2, 7, 9}):
                codes = codes | {7, 9}
            codes = codes | set((extra_codes or {}).get(col, []))
            series = series.mask(numeric.isin(codes))
            if not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors="coerce")
        elif (extra_codes or {}).get(col):
            series = series.mask(series.astype(str).isin({str(v) for v in extra_codes[col]}))
        cleaned[col] = series
        delta = int(series.isna().sum() - before)
        if delta:
            changes[str(col)] = delta
    return cleaned, changes


def split_dataset(df, method, pct=70, col=None, values=None, seed=42):
    """Turn ONE file into (baseline, production). See app sidebar for guidance."""
    if method == "random":
        b = df.sample(frac=pct / 100, random_state=seed)
        c = df.drop(b.index)
    elif method == "order":
        k = int(len(df) * pct / 100)
        b, c = df.iloc[:k], df.iloc[k:]
    elif method == "numeric":
        cut = pd.to_numeric(df[col], errors="coerce").quantile(pct / 100)
        v = pd.to_numeric(df[col], errors="coerce")
        b, c = df[v <= cut], df[v > cut]
    else:  # category
        m = df[col].astype(str).isin([str(v) for v in values])
        b, c = df[m], df[~m]
    return b.reset_index(drop=True), c.reset_index(drop=True)


# ------------------------------------------------------ schema logic ----
def guess_target(df: pd.DataFrame):
    binary = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    for hint in TARGET_HINTS:
        for c in binary:
            if hint in c.lower():
                return c
    return None


def is_id_like(s: pd.Series, name: str) -> bool:
    if re.search(r"(^|_)(id|uuid|mrn)$|^id(_|$)|patient_?id", name.lower()):
        return True
    n = s.nunique(dropna=True)
    if n == len(s) and (pd.api.types.is_integer_dtype(s) or not pd.api.types.is_numeric_dtype(s)):
        return True
    return (not pd.api.types.is_numeric_dtype(s)) and n > 50 and n > 0.9 * len(s)


def suggest_features(df: pd.DataFrame, target) -> list[str]:
    out = []
    for c in df.columns:
        s = df[c]
        if c == target or s.nunique(dropna=True) <= 1 or is_id_like(s, c):
            continue
        if not pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 50:
            continue  # free text / dates / high-cardinality
        out.append(c)
    return out


def classify(df: pd.DataFrame, cols, max_cat: int = 10) -> dict[str, str]:
    types = {}
    for c in cols:
        s = df[c]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        types[c] = "numeric" if numeric and s.nunique(dropna=True) > max_cat else "categorical"
    return types


def prepare(df: pd.DataFrame, feats, types, target=None) -> pd.DataFrame:
    out = {}
    for c in feats:
        if types[c] == "numeric":
            out[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        else:
            out[c] = df[c].astype(object).where(df[c].notna(), "Missing").astype(str).to_numpy()
    if target:
        out[target] = df[target].to_numpy()
    return pd.DataFrame(out)


def subsample(df: pd.DataFrame, n: int = 10000, seed: int = 0) -> pd.DataFrame:
    return df if len(df) <= n else df.sample(n, random_state=seed).reset_index(drop=True)


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


# --------------------------------------------------------- statistics ----
def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant column: compare by exact value instead
        return psi_from_counts(*cat_counts(pd.Series(ref).astype(str), pd.Series(cur).astype(str)))
    edges[0], edges[-1] = -np.inf, np.inf
    return psi_from_counts(np.histogram(ref, edges)[0], np.histogram(cur, edges)[0])


def psi_from_counts(rc, cc) -> float:
    r = np.clip(np.asarray(rc, float) / max(np.sum(rc), 1), 1e-4, None)
    c = np.clip(np.asarray(cc, float) / max(np.sum(cc), 1), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def cat_counts(ref: pd.Series, cur: pd.Series, top: int = 20):
    keep = set(ref.value_counts().index[:top])
    r = ref.where(ref.isin(keep), "Other")
    c = cur.where(cur.isin(keep), "Other")
    cats = sorted(set(r) | set(c))
    return (r.value_counts().reindex(cats, fill_value=0),
            c.value_counts().reindex(cats, fill_value=0))


def _row(name, kind, base_s, cur_s, warn, crit):
    if kind == "numeric":
        b, c = base_s.dropna().to_numpy(float), cur_s.dropna().to_numpy(float)
        if len(b) < 2 or len(c) < 2:
            return None
        ks = stats.ks_2samp(b, c)
        w = float(stats.wasserstein_distance(b, c))
        p = psi_numeric(b, c)
        shift = f"{100 * (c.mean() - b.mean()) / abs(b.mean()):+.1f}%" if b.mean() else "n/a"
        vals = dict(Baseline=f"{b.mean():.2f}", Current=f"{c.mean():.2f}", Shift=shift,
                    Test="KS", Statistic=ks.statistic, **{"p-value": ks.pvalue},
                    Distance=w / (b.std() or 1.0), **{"Distance Metric": "Wasserstein (σ)"})
    else:
        rc, cc = cat_counts(base_s, cur_s)
        rp, cp = rc / rc.sum(), cc / cc.sum()
        p = psi_from_counts(rc, cc)
        try:
            chi = stats.chi2_contingency(np.vstack([rc.values, cc.values]).T)
            stat, pv = float(chi[0]), float(chi[1])
        except ValueError:
            stat, pv = np.nan, np.nan
        d = (cp - rp)
        top = d.abs().idxmax()
        vals = dict(Baseline=f"{rp.idxmax()} ({rp.max():.0%})", Current=f"{cp.idxmax()} ({cp.max():.0%})",
                    Shift=f"{100 * d[top]:+.1f}pp {top}", Test="Chi²", Statistic=stat, **{"p-value": pv},
                    Distance=float(jensenshannon(rp.values, cp.values, base=2)),
                    **{"Distance Metric": "Jensen-Shannon"})
    return {"Feature": name, "Type": kind, **vals, "PSI": p,
            "Status": "Critical" if p >= crit else "Warning" if p >= warn else "Stable"}


def drift_table(base, cur, feats, types, warn=0.10, crit=0.25) -> pd.DataFrame:
    rows = [_row(f, types[f], base[f], cur[f], warn, crit) for f in feats]
    return pd.DataFrame([r for r in rows if r])


# ----------------------------------------------------- model health -----
def model_health(base, cur, feats, types, target) -> dict:
    """Fit a proxy model on baseline and score current data; not a deployed endpoint."""
    if not target or target not in base or target not in cur or base[target].nunique() != 2:
        return {}
    classes = sorted(base[target].dropna().unique(), key=str)
    mapping = {classes[0]: 0, classes[1]: 1}
    yb = base[target].map(mapping)
    if yb.isna().any():
        base, yb = base[yb.notna()], yb.dropna()
    num = [f for f in feats if types[f] == "numeric"]
    cat = [f for f in feats if types[f] == "categorical"]
    pre = ColumnTransformer([
        ("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("c", OneHotEncoder(handle_unknown="ignore"), cat)])
    model = make_pipeline(pre, LogisticRegression(max_iter=500)).fit(base[feats], yb.astype(int))
    pb, pc = model.predict_proba(base[feats])[:, 1], model.predict_proba(cur[feats])[:, 1]
    out = {"positive_class": str(classes[1]), "pred_psi": psi_numeric(pb, pc),
           "mean_risk_base": float(pb.mean()), "mean_risk_cur": float(pc.mean()),
           "prevalence_base": float(yb.mean())}
    yc = cur[target].map(mapping)
    ok = yc.notna()
    if ok.sum() > 30:
        out["prevalence_cur"] = float(yc[ok].mean())
        out["brier_base"] = float(brier_score_loss(yb.astype(int), pb))
        out["brier_cur"] = float(brier_score_loss(yc[ok].astype(int), pc[ok.to_numpy()]))
    return out


def audit_verdict(table, evidently_share, max_share, crit_psi):
    reasons = [f"{r.Feature}: PSI {r.PSI:.3f} ≥ critical threshold {crit_psi:.2f}"
               for _, r in table[table.Status == "Critical"].iterrows()]
    share = evidently_share if evidently_share is not None else (table.PSI >= 0.10).mean()
    if share >= max_share:
        reasons.append(f"Drifted-feature share {share:.0%} ≥ dataset limit {max_share:.0%}")
    return not reasons, reasons


COMPLIANCE_MAP = [
    ("FDA", "Good Machine Learning Practice (GMLP), Principle 10",
     "Deployed models are monitored for performance and re-training risks",
     "Per-feature PSI, KS/Chi² and distance metrics on every production cohort"),
    ("FDA", "Predetermined Change Control Plan (PCCP)",
     "Pre-specified triggers for model modification",
     "Critical PSI and drifted-share thresholds act as retraining triggers"),
    ("EU AI Act", "Art. 9, Risk management system",
     "Continuous, iterative risk identification across the lifecycle",
     "Drift-check result and severity grading per monitored feature"),
    ("EU AI Act", "Art. 10, Data and data governance",
     "Training data relevant and representative of the deployed population",
     "Baseline vs production distribution comparison"),
    ("EU AI Act", "Art. 15, Accuracy and robustness",
     "Consistent performance over the system lifetime",
     "Prediction drift (PSI) and Brier-score calibration check"),
    ("EU AI Act", "Art. 72, Post-market monitoring",
     "Systematic collection and analysis of performance data",
     "Timestamped audit log with dataset fingerprints and exportable report"),
    ("HIPAA", "45 CFR 164.312(b), Audit controls",
     "Record and examine activity in systems handling ePHI",
     "Run log; datasets identified by hash, not row content"),
]


# -------------------------------------------------------- sample data ----
def _generate(n, seed, male_p=0.5, smoke_p=(0.60, 0.25, 0.15)):
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(52, 12, n), 18, 90)
    bmi = np.clip(rng.normal(28 + 0.03 * (age - 50), 4.5, n), 16, 55)
    glucose = np.clip(rng.normal(98 + 0.8 * (bmi - 28) + 0.15 * (age - 50), 14, n), 60, 300)
    return pd.DataFrame({
        "age": age, "sex": rng.choice(["Male", "Female"], n, p=[male_p, 1 - male_p]),
        "smoking": rng.choice(["Never", "Former", "Current"], n, p=smoke_p),
        "bmi": bmi, "fasting_glucose": glucose,
        "hba1c": np.clip(4.6 + 0.0125 * glucose + rng.normal(0, 0.25, n), 4.0, 14.0),
        "blood_pressure": np.clip(rng.normal(112 + 0.25 * (age - 50) + 0.5 * (bmi - 28), 12, n), 80, 220),
        "cholesterol": np.clip(rng.normal(195, 35, n), 100, 400)})


def _label(df, seed):
    z = (-12.0 + 0.02 * df.age + 0.06 * df.bmi + 0.03 * df.fasting_glucose + 0.55 * df.hba1c
         + 0.01 * df.blood_pressure + 0.002 * df.cholesterol)
    return (np.random.default_rng(seed + 99).random(len(df)) < 1 / (1 + np.exp(-z))).astype(int)


def generate_baseline(n=5000, seed=42):
    df = _generate(n, seed)
    df["diabetes_onset"] = _label(df, seed)
    return df.round(2)


def generate_production(n=2000, seed=7, level=1.0):
    p = np.array([0.60, 0.25, 0.15]) + level * np.array([-0.08, -0.02, 0.10])
    df = _generate(n, seed, male_p=0.5 + 0.08 * level, smoke_p=tuple(p / p.sum()))
    df["age"] += 4.0 * level
    df["bmi"] += 1.5 * level
    df["fasting_glucose"] = (df.fasting_glucose - 98) * (1 + 0.15 * level) + 98 + 7 * level
    df["hba1c"] += 0.05 * level
    df["blood_pressure"] += 3.5 * level
    df["cholesterol"] += 12 * level
    df["diabetes_onset"] = _label(df, seed)
    return df.round(2)