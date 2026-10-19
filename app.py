from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import utils as U

st.set_page_config(page_title="SynthetixMed AI Governance", page_icon="🩺", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --slate:#475569; --card:rgba(100,116,139,.10); --line:rgba(100,116,139,.28); }
html, body, [class*="css"], .stApp { font-family:'Inter','Roboto','SF Pro Display',-apple-system,'Segoe UI',sans-serif; }
@media (prefers-color-scheme: dark) {
  .stApp { background:linear-gradient(160deg,#0B132B 0%,#1C2541 35%,#0a0a0f 100%); }
  [data-testid="stSidebar"] { background:linear-gradient(180deg,#0B132B,#10101a); }
}
[data-testid="stDecoration"] { display:none; }
.block-container { padding-top:4.5rem; max-width:1400px; }
h1,h2,h3 { letter-spacing:-.01em; font-weight:650; }
.hero { background:linear-gradient(120deg,#0B132B 0%,#1C2541 100%); border-radius:12px;
        padding:1.2rem 1.6rem; margin-bottom:1rem; border:1px solid rgba(255,255,255,.08); }
.hero .t { font-size:1.5rem; font-weight:700; color:#FFFFFF; letter-spacing:-.01em; }
.hero p { color:#CBD5E1; margin:.25rem 0 0 0; font-size:.95rem; }
.badge { border-radius:10px; padding:1rem 1.4rem; color:#fff; font-weight:700; font-size:1.2rem;
         border:1px solid rgba(255,255,255,.18); margin-bottom:.4rem; }
.badge small { display:block; font-weight:500; font-size:.85rem; opacity:.92; margin-top:.25rem; }
.badge.pass { background:linear-gradient(90deg,#14532D,#16A34A); box-shadow:0 0 22px rgba(22,163,74,.35); }
.badge.fail { background:linear-gradient(90deg,#7F1D1D,#DC2626); box-shadow:0 0 22px rgba(220,38,38,.45); }
.kpi { border-left:4px solid #2563EB; background:var(--card); border:1px solid var(--line);
       border-left:4px solid #2563EB; border-radius:10px; padding:.9rem 1.1rem; height:100%; }
.kpi .v { font-size:1.7rem; font-weight:700; line-height:1.2; }
.kpi .l { color:var(--slate); font-size:.85rem; }
.kpi .n { font-size:.8rem; margin-top:.2rem; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
.stTabs [aria-selected="true"] { color:#2563EB; }
.stTabs [data-baseweb="tab-highlight"] { background-color:#2563EB; }
</style>
""", unsafe_allow_html=True)

COLORS = {"Stable": "#16A34A", "Warning": "#D97706", "Critical": "#DC2626"}
PLOT = dict(template="plotly", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"), margin=dict(l=10, r=10, t=40, b=10))
METHODS = {"Random split (control test: should PASS)": "random",
           "By row order (oldest vs newest rows)": "order",
           "By a numeric column (e.g. age)": "numeric",
           "By a category (e.g. hospital, sex, region)": "category"}
METHOD_HELP = {
    "random": "Both halves come from the same population, so a healthy pipeline should PASS. "
              "Use this to sanity-check the tool.",
    "order": "Use when rows are sorted by time. Earliest rows = baseline, latest rows = production.",
    "numeric": "Simulates a population shift, e.g. train on younger patients, deploy on older ones. "
               "The split column itself will always show critical drift by construction; "
               "look at the OTHER columns.",
    "category": "Simulates deploying to a new group, e.g. train on Hospital A, deploy at Hospital B. "
                "The split column itself will always drift by construction."}


# ---------------------------------------------------------------- cache ---
@st.cache_data(show_spinner=False)
def load(raw: bytes, filename: str):
    return U.read_dataset(raw, filename)


@st.cache_data(show_spinner=False)
def sample_data(n_base, n_cur, level):
    return U.generate_baseline(n_base), U.generate_production(n_cur, level=level)


@st.cache_data(show_spinner="Computing drift statistics…")
def cached_table(base, cur, feats, types, warn, crit):
    return U.drift_table(base, cur, feats, types, warn, crit)


@st.cache_data(show_spinner="Scoring reference risk model…")
def cached_health(base, cur, feats, types, target):
    return U.model_health(base, cur, list(feats), types, target)


@st.cache_data(show_spinner="Running Evidently…")
def evidently_report(base, cur, feats, types, share):
    """Evidently DataDriftPreset + DatasetDriftMetric on a capped sample. -> (html, summary, error)"""
    try:
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.metrics import DatasetDriftMetric
        from evidently.report import Report

        b, c = U.subsample(base[list(feats)]), U.subsample(cur[list(feats)])
        cm = ColumnMapping(numerical_features=[f for f in feats if types[f] == "numeric"],
                           categorical_features=[f for f in feats if types[f] == "categorical"])
        rep = Report(metrics=[DataDriftPreset(drift_share=share), DatasetDriftMetric(drift_share=share)])
        rep.run(reference_data=b, current_data=c, column_mapping=cm)
        html = rep.get_html() if hasattr(rep, "get_html") else rep.as_html()
        res = next(m["result"] for m in rep.as_dict()["metrics"] if m["metric"] == "DatasetDriftMetric")
        return html, {"share": float(res["share_of_drifted_columns"]),
                      "n_drifted": int(res["number_of_drifted_columns"]),
                      "n_cols": int(res["number_of_columns"]),
                      "dataset_drift": bool(res["dataset_drift"]),
                      "rows_used": [len(b), len(c)]}, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"

# sidebar 
raw_b = raw_c = None
cleaning_summary = {}
feats, target, types = [], None, {}
with st.sidebar:
    st.markdown("### 🩺 SynthetixMed")
    st.caption("AI Governance Console · v1.1")
    model_name = st.text_input("Monitored model / condition", "Type 2 Diabetes onset model")
    source = st.radio("Data source", ["Built-in sample data", "Two CSV files", "One CSV (split it)"])

    if source == "Built-in sample data":
        scen = st.selectbox("Production scenario", list(U.SCENARIOS), index=0)
        n_base = st.slider("Baseline rows", 1000, 20000, 5000, 500)
        n_cur = st.slider("Production rows", 500, 10000, 2000, 500)
        raw_b, raw_c = sample_data(n_base, n_cur, U.SCENARIOS[scen])
    elif source == "Two CSV files":
        fb = st.file_uploader("Baseline Training Dataset", type=["csv", "xpt"], key="b", help="Upload CSV or extracted SAS Transport (.xpt) file. ZIP archives must be extracted first.")
        fc = st.file_uploader("Current Production Dataset", type=["csv", "xpt"], key="c", help="Upload CSV or extracted SAS Transport (.xpt) file.")
        if fb and fc:
            try:
                raw_b, raw_c = load(fb.getvalue(), fb.name), load(fc.getvalue(), fc.name)
            except (ValueError, OSError, ImportError) as exc:
                st.error(f"Unable to read uploaded dataset: {exc}")
                st.stop()
    else:
        f1 = st.file_uploader("Full dataset (CSV or XPT)", type=["csv", "xpt"], key="one", help="Upload CSV or extracted SAS Transport (.xpt) file.")
        if f1:
            try:
                full = load(f1.getvalue(), f1.name)
            except (ValueError, OSError, ImportError) as exc:
                st.error(f"Unable to read uploaded dataset: {exc}")
                st.stop()
            label = st.selectbox("How to split it", list(METHODS))
            method = METHODS[label]
            st.caption(METHOD_HELP[method])
            kw = {}
            if method in ("random", "order"):
                kw["pct"] = st.slider("% of rows used as baseline", 30, 90, 70)
            elif method == "numeric":
                cols = [c for c in full.columns if pd.api.types.is_numeric_dtype(full[c])]
                kw["col"] = st.selectbox("Split on column", cols)
                kw["pct"] = st.slider("Lowest X% of values = baseline", 30, 90, 60)
            else:
                cols = [c for c in full.columns if full[c].nunique() <= 50]
                kw["col"] = st.selectbox("Split on column", cols)
                opts = sorted(full[kw["col"]].astype(str).unique())
                kw["values"] = st.multiselect("Values that form the BASELINE", opts, default=opts[:max(1, len(opts) // 2)])
            if method != "category" or kw["values"]:
                raw_b, raw_c = U.split_dataset(full, method, **kw)
            else:
                st.warning("Pick at least one baseline value.")

    if raw_b is not None:
        st.markdown("**Data hygiene**")
        st.caption("Conservative sentinel cleaning is applied to both cohorts. Codes 7/9 are only automatically cleaned for recognizable survey-style binary flags; use the source codebook for other fields. Original uploads are not modified.")
        with st.expander("Optional missing-code overrides", expanded=False):
            st.caption("Add column-specific codes using `column:code1,code2` per line (e.g. `AGE_GROUP:77,99`). Never apply 7 or 9 to all columns.")
            override_text = st.text_area("Per-column missing codes", value="", key="missing_overrides")
        overrides = {}
        try:
            for line in override_text.splitlines():
                if not line.strip():
                    continue
                col, codes = line.split(":", 1)
                col = col.strip()
                if col not in raw_b.columns and col not in raw_c.columns:
                    raise ValueError(f"Unknown column: {col}")
                overrides[col] = [float(v.strip()) for v in codes.split(",") if v.strip()]
        except (ValueError, TypeError) as exc:
            st.error(f"Invalid missing-code override: {exc}")
            st.stop()
        raw_b, cleaned_b = U.clean_missing_codes(raw_b, overrides)
        raw_c, cleaned_c = U.clean_missing_codes(raw_c, overrides)
        cleaning_summary = {"baseline": cleaned_b, "current": cleaned_c}
        st.caption(f"Sentinel values mapped to missing: baseline {sum(cleaned_b.values()):,}, current {sum(cleaned_c.values()):,}. Missingness is retained as a category for categorical drift; numeric tests exclude NaNs.")
        shared = [c for c in raw_b.columns if c in raw_c.columns]
        only = sorted(set(raw_b.columns) ^ set(raw_c.columns))
        if only:
            st.warning("Columns not in both files are ignored: " + ", ".join(only))
        st.divider()
        st.markdown("**Columns**")
        guess = U.guess_target(raw_b[shared]) if shared else None
        t_opts = ["(none)"] + shared
        t_sel = st.selectbox("Outcome / label column (optional)", t_opts,
                             index=t_opts.index(guess) if guess else 0,
                             help="Enables baseline-trained reference-model simulation metrics, not evaluation of a connected deployed model. Requires a binary outcome.")
        target = None if t_sel == "(none)" else t_sel
        max_cat = st.slider("Treat numeric columns with ≤ N unique values as categories", 2, 20, 10)
        feats = st.multiselect("Columns to monitor", shared, default=U.suggest_features(raw_b[shared], target))
    st.divider()
    st.markdown("**Safety thresholds**")
    warn_psi = st.slider("PSI warning", 0.05, 0.25, 0.10, 0.01)
    crit_psi = st.slider("PSI critical (fail)", 0.15, 0.50, 0.25, 0.01)
    max_share = st.slider("Max drifted-feature share", 0.1, 1.0, 0.33, 0.01)

st.markdown(f'<div class="hero"><div class="t">SynthetixMed AI Governance &amp; Clinical Data Drift Dashboard</div>'
            f'<p>{model_name} · post-deployment monitoring for FDA, EU AI Act and HIPAA audit readiness</p></div>',
            unsafe_allow_html=True)

if raw_b is None:
    st.info("Choose a data source in the sidebar. With **One CSV (split it)** you upload a single file and the "
            "tool divides it into baseline and production for you.")
    st.stop()
if not feats:
    st.warning("Select at least one column to monitor.")
    st.stop()
if min(len(raw_b), len(raw_c)) < 30:
    st.error(f"Each dataset needs at least 30 rows (baseline {len(raw_b)}, production {len(raw_c)}).")
    st.stop()

types = U.classify(raw_b, feats, max_cat)
base = U.prepare(raw_b, feats, types, target)
cur = U.prepare(raw_c, feats, types, target)
ftuple, ttuple = tuple(feats), types

table = cached_table(base, cur, ftuple, ttuple, warn_psi, crit_psi)
html, ev, ev_err = evidently_report(base, cur, ftuple, ttuple, max_share)
health = cached_health(base, cur, ftuple, ttuple, target)
passed, reasons = U.audit_verdict(table, ev["share"] if ev else None, max_share, crit_psi)
max_psi = table.PSI.max()
now = datetime.now(timezone.utc)
n_num = sum(t == "numeric" for t in types.values())

report = {
    "generated_utc": now.isoformat(timespec="seconds"), "model": model_name,
    "verdict": "DRIFT CHECK PASSED" if passed else "DRIFT CHECK FAILED", "reasons": reasons,
    "thresholds": {"psi_warning": warn_psi, "psi_critical": crit_psi, "max_drifted_share": max_share},
    "monitored_columns": types, "outcome_column": target,
    "datasets": {"baseline": {"rows": len(base), "sha": U.fingerprint(base[feats])},
                 "production": {"rows": len(cur), "sha": U.fingerprint(cur[feats])}},
    "evidently": ev, "reference_model_simulation": health, "missing_code_cleaning": cleaning_summary,
    "scope": "Automated drift-threshold check only; not a clinical safety assessment or regulatory certification", "features": json.loads(table.to_json(orient="records")),
}
fp = (report["datasets"]["baseline"]["sha"], report["datasets"]["production"]["sha"],
      warn_psi, crit_psi, max_share, ftuple)
log = st.session_state.setdefault("audit_log", [])
if st.session_state.get("last_fp") != fp:
    st.session_state["last_fp"] = fp
    log.append({"Timestamp (UTC)": now.strftime("%Y-%m-%d %H:%M:%S"), "Model": model_name,
                "Verdict": report["verdict"], "Max PSI": round(max_psi, 3),
                "Drifted share": f"{(ev['share'] if ev else 0):.0%}",
                "Baseline SHA": fp[0], "Production SHA": fp[1],
                "Thresholds": f"warn {warn_psi} / crit {crit_psi} / share {max_share}"})

if passed:
    st.markdown('<div class="badge pass">✅ DRIFT CHECK PASSED: Configured Drift Thresholds Met'
                f'<small>No configured drift guardrails were breached; max PSI {max_psi:.3f}. This does not establish clinical safety.</small></div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="badge fail">🚨 DRIFT CHECK FAILED: Drift Threshold Exceeded'
                f'<small>{len(reasons)} guardrail breach(es). Review the Evidently Drift Report and escalate to clinical governance.</small></div>',
                unsafe_allow_html=True)
st.caption(f"Monitoring {len(feats)} columns ({n_num} numeric, {len(feats) - n_num} categorical) · "
           f"{len(base):,} baseline rows vs {len(cur):,} production rows"
           + (f" · outcome column: {target}" if target else ""))

tab1, tab2, tab3, tab4 = st.tabs(["Overview & KPI Summary", "Evidently Drift Report",
                                  "Feature Distributions", "Governance Monitoring Logs"])


def kpi(col, label, value, note="", color=None):
    style = f' style="color:{color}"' if color else ""
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v"{style}>{value}</div>'
                 f'<div class="n">{note}</div></div>', unsafe_allow_html=True)


with tab1:
    n_crit, n_warn = int((table.Status == "Critical").sum()), int((table.Status == "Warning").sum())
    c = st.columns(5)
    kpi(c[0], "Critical features", f"{n_crit} / {len(table)}", f"{n_warn} on warning",
        COLORS["Critical"] if n_crit else COLORS["Stable"])
    kpi(c[1], "Max PSI", f"{max_psi:.3f}", table.loc[table.PSI.idxmax(), "Feature"])
    kpi(c[2], "Evidently drifted share", f"{ev['share']:.0%}" if ev else "n/a", f"limit {max_share:.0%}")
    kpi(c[3], "Simulated risk-score PSI", f"{health['pred_psi']:.3f}" if health else "n/a",
        "baseline-trained proxy model" if health else "needs a 2-class outcome column")
    if "brier_cur" in health:
        kpi(c[4], "Simulated Brier score", f"{health['brier_cur']:.4f}", f"baseline {health['brier_base']:.4f}")
    else:
        kpi(c[4], "Cohort sizes", f"{len(cur):,}", f"baseline {len(base):,}")
    st.caption("Reference-Model Simulation Performance: a logistic regression proxy is trained on the baseline cohort and scored on the current cohort. No deployed model endpoint is connected; these are not validated clinical performance results.")
    st.write("")
    t = table.sort_values("PSI")
    fig = go.Figure(go.Bar(x=t.PSI, y=t.Feature, orientation="h", marker_color=[COLORS[s] for s in t.Status],
                           text=t.PSI.round(3), textposition="outside"))
    fig.add_vline(x=warn_psi, line_dash="dot", line_color=COLORS["Warning"], annotation_text="warning")
    fig.add_vline(x=crit_psi, line_dash="dash", line_color=COLORS["Critical"], annotation_text="critical")
    fig.update_layout(title="Population Stability Index by feature", xaxis_title="PSI",
                      height=max(320, 48 * len(t) + 120), **PLOT)
    st.plotly_chart(fig, use_container_width=True)
    if reasons:
        st.markdown("**Guardrail breaches**")
        for r in reasons:
            st.markdown(f"- {r}")
    st.dataframe(table.style.format({"Statistic": "{:.3f}", "p-value": "{:.2e}", "PSI": "{:.3f}",
                                     "Distance": "{:.3f}"}, na_rep="–")
                 .map(lambda s: f"color:{COLORS[s]};font-weight:700" if s in COLORS else "", subset=["Status"]),
                 use_container_width=True, hide_index=True)

with tab2:
    if ev_err:
        st.error(f"Evidently could not run ({ev_err}). The scipy/numpy engine remains authoritative. "
                 "Check that `evidently==0.4.40` is installed.")
    else:
        c = st.columns(3)
        kpi(c[0], "Dataset drift (Evidently)", "DETECTED" if ev["dataset_drift"] else "Not detected", "",
            COLORS["Critical"] if ev["dataset_drift"] else COLORS["Stable"])
        kpi(c[1], "Drifted columns", f"{ev['n_drifted']} / {ev['n_cols']}")
        kpi(c[2], "Share of drifted columns", f"{ev['share']:.0%}", f"threshold {max_share:.0%}")
        st.caption(f"Evidently runs on a random sample of up to 10,000 rows per dataset "
                   f"({ev['rows_used'][0]:,} baseline / {ev['rows_used'][1]:,} production used) to stay fast.")
        components.html(html, height=1000, scrolling=True)
        st.download_button("Download Evidently HTML report", html, "evidently_drift_report.html", "text/html")

with tab3:
    key = st.selectbox("Feature", feats)
    row = table[table.Feature == key].iloc[0]
    if types[key] == "numeric":
        b, cu = base[key].dropna(), cur[key].dropna()
        l, r = st.columns(2)
        h = go.Figure()
        h.add_histogram(x=b, name="Baseline", histnorm="probability density", opacity=.6, marker_color="#3B82F6", nbinsx=40)
        h.add_histogram(x=cu, name="Production", histnorm="probability density", opacity=.6, marker_color="#F59E0B", nbinsx=40)
        h.update_layout(barmode="overlay", title=f"{key}: density", height=380, **PLOT)
        l.plotly_chart(h, use_container_width=True)
        e = go.Figure()
        for name, s, col in (("Baseline", b, "#3B82F6"), ("Production", cu, "#F59E0B")):
            xs = np.sort(s.to_numpy())
            idx = np.linspace(0, len(xs) - 1, min(len(xs), 2000)).astype(int)
            e.add_scatter(x=xs[idx], y=(idx + 1) / len(xs), name=name, mode="lines", line=dict(color=col, width=2))
        e.update_layout(title="Empirical CDF (KS = max vertical gap)", height=380, **PLOT)
        r.plotly_chart(e, use_container_width=True)
    else:
        rc, cc = U.cat_counts(base[key], cur[key])
        f = go.Figure([go.Bar(name="Baseline", x=list(rc.index), y=rc / rc.sum(), marker_color="#3B82F6"),
                       go.Bar(name="Production", x=list(cc.index), y=cc / cc.sum(), marker_color="#F59E0B")])
        f.update_layout(barmode="group", title=f"{key}: category share", yaxis_tickformat=".0%", height=380, **PLOT)
        st.plotly_chart(f, use_container_width=True)
    c = st.columns(4)
    kpi(c[0], "PSI", f"{row.PSI:.3f}", row.Status, COLORS[row.Status])
    kpi(c[1], f"{row.Test} statistic", f"{row.Statistic:.3f}", f"p = {row['p-value']:.2e}")
    kpi(c[2], row["Distance Metric"], f"{row.Distance:.3f}")
    kpi(c[3], "Shift", row.Shift, f"{row.Baseline} → {row.Current}")

with tab4:
    st.subheader("Governance framework mapping (illustrative)")
    st.warning("DEMONSTRATION ONLY: Results are automated statistical drift checks against configurable thresholds, not clinical diagnoses, patient-safety determinations, FDA clearance, EU AI Act compliance assessments, or HIPAA certification. Governance mapping is illustrative, not legal advice.")
    cmap = pd.DataFrame(U.COMPLIANCE_MAP, columns=["Framework", "Requirement", "Intent", "Control in this system"])
    cmap["Current evidence"] = f"{report['verdict']}; max PSI {max_psi:.3f}"
    st.dataframe(cmap, use_container_width=True, hide_index=True)
    st.subheader("Audit run log")
    st.caption("Session-scoped. Datasets are recorded by hash only, so no patient data enters the log. "
               "Persist to write-once storage in production.")
    st.dataframe(pd.DataFrame(log[::-1]), use_container_width=True, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button("Download drift governance report (JSON)", json.dumps(report, indent=2),
                       f"drift_governance_report_{now:%Y%m%dT%H%M%S}.json", "application/json")
    d2.download_button("Download monitoring log (CSV)", pd.DataFrame(log).to_csv(index=False), "audit_log.csv", "text/csv")