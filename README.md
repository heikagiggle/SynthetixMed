# SynthetixMed: Enterprise Clinical AI Data Drift & Governance Dashboard

> An enterprise-grade AI monitoring and regulatory compliance dashboard built for HealthTech/MedTech. Designed to audit clinical machine learning diagnostics in real-time, enforce automated safety guardrails, and satisfy regulatory requirements (EU AI Act / FDA post-market surveillance).

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)
![Evidently AI](https://img.shields.io/badge/Evidently-AI-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🏥 Clinical Problem & Governance Angle
In healthcare, machine learning models predicting conditions like **Type 2 Diabetes onset** are vulnerable to silent failure caused by **covariate shift** (changing patient demographics, shifting laboratory assay standards, or evolving socioeconomic distributions). 

If a diagnostic model is trained on one hospital cohort and deployed to another without monitoring, its predictive accuracy can plummet dangerously. **SynthetixMed** solves this by bridging the gap between data science and clinical risk management:
- **Automated Guardrails:** Continuously tracks demographic and physiological feature drift using statistical tests and Evidently AI.
- **Auditor-Ready Compliance:** Provides instant green/red compliance indicators and detailed metric reports that prove to regulatory auditors your system actively flags degraded AI reliability.

---

## 📊 Key Features
- **Interactive File Uploads:** Upload your own Baseline (training) and Current (production inference) CSV datasets, or use the built-in synthetic clinical generator.
- **Evidently AI Integration:** Deep integration with industry-standard Evidently metrics for automated data drift presets.
- **Dynamic Risk Badging:** Instant visual feedback displaying regulatory compliance status (`AUDIT PASSED` vs `AUDIT FAILED`).
- **Clinical Feature Breakdown:** Granular violin and distribution plots for critical biometric indicators (BMI, HbA1c, Blood Glucose, Age, Blood Pressure).

---

## 🛠️ Tech Stack
- **Frontend & Dashboard:** Streamlit (Custom styled for dark/light medical enterprise aesthetics)
- **Monitoring & Governance:** Evidently AI
- **Data Processing & Statistics:** Pandas, NumPy, SciPy (Kolmogorov-Smirnov & Population Stability Index)
- **Visualization:** Plotly & Evidently native HTML components

---

## 🚀 Installation & Local Running Guide

Follow these steps to set up and run the dashboard locally on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/synthetix-med-dashboard.git
cd synthetix-med-dashboard
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
```

**On macOS / Linux:**

```bash
source venv/bin/activate
```

**On Windows PowerShell:**

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

Open your browser at [http://localhost:8501](http://localhost:8501) to interact with the live dashboard!