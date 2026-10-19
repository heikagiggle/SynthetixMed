# SynthetixMed: Enterprise Clinical AI Data Drift & Governance Dashboard

An interactive clinical AI post-deployment monitoring and data drift governance dashboard built with Python, Streamlit, and Evidently AI.

SynthetixMed demonstrates how statistical monitoring, feature drift detection, data quality checks, reference-model simulations, and automated governance reporting can support oversight of machine learning systems in healthcare.

Designed as a HealthTech/MedTech portfolio project exploring monitoring practices relevant to FDA post-market oversight, the EU AI Act, and HIPAA-aware data governance. It is not a clinical diagnostic tool or regulatory certification engine.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)
![Evidently AI](https://img.shields.io/badge/Evidently-AI-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## Clinical Problem & Governance Angle
Machine learning systems used in healthcare can experience changes in their input data after deployment. These changes, known as **data drift or covariate shift** may arise from differences in patient demographics, clinical measurements, data collection practices, or healthcare populations.

For applications such as Type 2 Diabetes risk prediction, monitoring these changes is an important part of maintaining visibility into model behavior over time. However, detecting data drift alone does not establish a reduction in predictive performance.

**SynthetixMed** demonstrates a post-deployment monitoring workflow through statistical drift detection, reference-model simulations, data quality assessments, and automated governance reporting.

If a diagnostic model is trained on one hospital cohort and deployed to another without monitoring, its predictive accuracy can plummet dangerously. **SynthetixMed** solves this by bridging the gap between data science and clinical risk management:
- **Automated Drift Monitoring:** Evaluates demographic and clinical feature distributions using statistical tests and Evidently AI to identify potential changes between baseline and monitoring populations.
- **Governance Reporting & Traceability:** Generates threshold-based drift check statuses, statistical summaries, and audit logs to demonstrate how automated monitoring can support clinical AI oversight and post-deployment governance workflows.

---

## Key Features
- **Interactive File Uploads:** Upload your own Baseline (training) and Current (production inference) CSV and SAS XPT datasets, with built-in synthetic data and single-file splitting.
- **Evidently AI Integration:** Deep integration with industry-standard Evidently metrics for automated data drift presets.
- **Drift Check Status:** Threshold-based monitoring indicators (DRIFT CHECK PASSED / DRIFT CHECK FAILED)
- **Clinical Feature Breakdown:** Granular violin and distribution plots for critical biometric indicators (BMI, HbA1c, Blood Glucose, Age, Blood Pressure).

---

## Test with Real-World CDC Datasets

SynthetixMed supports testing with real-world public health datasets from the **Centers for Disease Control and Prevention (CDC)**.

Two sampled datasets from the **Behavioral Risk Factor Surveillance System (BRFSS)** are provided to demonstrate how changes in population characteristics can be detected across different data collection periods.

### Download Test Datasets

| Dataset | Description | Download |
|---|---|---|
| CDC BRFSS 2023 | Historical baseline reference cohort | [Download CSV](sample_datasets/cdc_2023_diabetes_sample.csv) |
| CDC BRFSS 2024 | Later monitoring cohort (production proxy) | [Download CSV](sample_datasets/cdc_2024_diabetes_sample.csv) |

Both datasets contain 10,000 sampled survey records, with selected diabetes-related demographic, behavioral, and health indicators.

### How to Test the Dashboard

1. Download both CSV files using the links above.
2. Launch the SynthetixMed dashboard locally or through its deployed Streamlit URL.
3. Select **Two CSV files** from the sidebar.
4. Upload `cdc_2023_diabetes_sample.csv` as the **Baseline Training Dataset**.
5. Upload `cdc_2024_diabetes_sample.csv` as the **Current Production Dataset**.
6. Explore feature drift, distribution changes, statistical monitoring results, and governance check reports.

### What This Demonstrates

- **Population Drift:** Detect distribution changes between the 2023 and 2024 survey cohorts.
- **Statistical Monitoring:** Evaluate feature changes using PSI, KS, chi-square, and Evidently AI.
- **Data Quality:** Identify missing values and differences in shared feature distributions.
- **Reference-Model Simulation:** Explore model performance using a locally trained proxy model where suitable outcome data is available.
- **Governance Reporting:** Generate threshold-based drift monitoring summaries and audit logs.

**Data source:** CDC Behavioral Risk Factor Surveillance System (BRFSS), 2023 and 2024.

**Important:** The CDC datasets are real public health survey records, not data from a deployed clinical prediction system. The 2024 cohort serves as a simulated production monitoring population. Results should not be interpreted as clinical diagnoses, real-world model safety validation, or regulatory certification.

----

## Tech Stack
- **Frontend & Dashboard:** Streamlit (Custom styled for dark/light medical enterprise aesthetics)
- **Monitoring & Governance:** Evidently AI
- **Data Processing & Statistics:** Pandas, NumPy, SciPy (Kolmogorov-Smirnov & Population Stability Index)
- **Visualization:** Plotly & Evidently native HTML components

---

##  Installation & Local Running Guide

Follow these steps to set up and run the dashboard locally on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/heikagiggle/SynthetixMed.git
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
.\venv\Scripts\Activate.ps1
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