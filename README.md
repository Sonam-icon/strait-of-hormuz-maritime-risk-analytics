# Strait of Hormuz 2026 — Maritime Trade & Risk Analytics

> **Simulation-based Data Analytics / Data Science Project**

## Project Overview

This project analyzes a simulated 2026 Strait of Hormuz maritime crisis dataset to understand how different trade-access conditions are associated with:

- maritime rerouting
- transit cost
- insurance cost
- operational delays
- additional fuel consumption
- commodity exposure
- payment rails
- total asset value at risk

The project combines **Python, SQL-oriented analytics, statistical testing, machine learning, anomaly detection and Power BI-ready outputs**.

### Important data note

The supplied dataset describes itself as a **high-fidelity simulation**. Therefore, this project does **not** present the dataset's geopolitical narrative as independently verified historical fact. All findings should be interpreted as patterns within the supplied simulation.

---

# Business Problem

A maritime disruption can affect a vessel through several cost and operational channels:

```text
Trade Access
     │
     ├── Toll
     ├── Insurance Premium
     ├── Rerouting
     ├── Additional Fuel
     └── Delay
             │
             ▼
       Total Transit Cost
             │
             ▼
      Supply Chain Exposure
```

### Key analytical questions

1. How do transit costs differ across trade tiers?
2. What factors are associated with vessel rerouting?
3. How large is the simulated asset value exposed to disruption?
4. Which continents, commodities and payment rails show different risk patterns?
5. Are there statistically significant relationships between trade tier and operational outcomes?
6. Can simulated rerouting decisions be predicted using vessel/trade characteristics?
7. Can abnormal operational periods be detected automatically?

---

# Dataset

The project uses the following Kaggle dataset:

**Strait of Hormuz 2026: The Great Decoupling**

**Source:**  
https://www.kaggle.com/datasets/moaz1911/strait-of-hormuz-2026-the-great-decoupling

The raw CSV is **not included in this GitHub repository** because the dataset's redistribution/license terms have not been independently confirmed.

The original analysis was performed on the downloaded CSV supplied by the dataset publisher.

### Dataset characteristics used in the analysis

- **1,890 records**
- **342 unique vessels**
- Date range: **1 March 2026 – 31 May 2026**
- Trade tiers: Privileged, Taxed, Blocked

Important variables include:

| Category | Variables |
|---|---|
| Trade access | `trade_tier`, `transit_status`, `rerouted` |
| Financial friction | `toll_usd`, `insurance_cost_usd`, `reroute_penalty_usd`, `total_transit_cost_usd` |
| Operations | `days_delayed`, `extra_fuel_tonnes` |
| Exposure | `ship_hull_value_usd`, `estimated_cargo_value_usd`, `total_asset_value_at_risk_usd` |
| Geographical | `flag`, `continent`, `destination` |
| Commercial | `commodity`, `payment_rail` |
| Security | `naval_escort_status` |

---

# Project Architecture

```text
                         RAW CSV
                            │
                            ▼
                   DATA QUALITY CHECKS
                            │
                            ▼
                         EDA
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
          COST ANALYSIS  RISK ANALYSIS  ROUTING
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                   STATISTICAL TESTING
                            │
                            ▼
                  MACHINE LEARNING
                    /           \
                   /             \
          Logistic Regression   XGBoost
                   \             /
                    \           /
                     ▼         ▼
                  ANOMALY DETECTION
                            │
                            ▼
                    POWER BI OUTPUTS
```

---

# Analytical Workflow

## 1. Data Quality

The pipeline checks:

- missing values
- data types
- unique values
- date validity
- asset-value arithmetic consistency
- transit-cost arithmetic consistency

Two important reconciliation checks are:

```text
total_asset_value_at_risk
=
ship_hull_value
+
estimated_cargo_value
```

and

```text
total_transit_cost
=
toll
+
insurance_cost
+
reroute_penalty
```

The supplied dataset produced **zero mismatches** for both checks.

---

# 2. Exploratory Data Analysis

The analysis examines:

### Trade Tier

- vessel/record distribution
- average transit cost
- average delay
- rerouting rate
- insurance cost
- fuel penalty
- asset exposure

### Payment Rail

Comparison of simulated payment mechanisms such as:

- USD
- e-CNY
- CIPS
- USDT

### Geography

Comparison by:

- continent
- flag
- destination

### Commodity

Comparison across simulated cargo categories.

---

# 3. Statistical Analysis

### Test 1 — Transit Cost Difference

Question:

> Is there evidence that simulated Taxed and Privileged vessels have different transit costs?

Tests:

- Welch's independent-samples t-test
- Mann–Whitney U test

The analysis reports:

- test statistic
- p-value
- sample interpretation

### Test 2 — Trade Tier and Rerouting

A chi-square test of independence evaluates whether:

```text
trade_tier
      ×
rerouting
```

are statistically associated in the supplied simulation.

**Statistical association should not be interpreted as proof of causation.**

---

# 4. Rerouting Prediction

## Target

```text
reroute_flag
```

where:

```text
1 = rerouted
0 = not rerouted
```

## Predictors

The model uses:

- trade tier
- vessel flag
- commodity
- destination
- payment rail
- continent
- ship hull value
- estimated cargo value
- insurance premium delta

### Models

Two models are benchmarked:

1. Logistic Regression
2. XGBoost

### Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

A chronological train/test split is used rather than randomly mixing dates.

### Leakage Prevention

The following variables are deliberately excluded from the predictive model:

- `transit_status`
- `days_delayed`
- `reroute_penalty_usd`
- `total_transit_cost_usd`

These variables either directly encode the outcome or occur after the routing decision, so including them would create **target leakage**.

---

# 5. Anomaly Detection

Daily operational metrics are created for:

- vessel count
- rerouting rate
- average delay
- total transit cost
- total asset value at risk
- additional fuel consumption

Two anomaly approaches are implemented:

### Rolling Z-score

A 30-day rolling baseline is used to identify unusually high/low daily transit cost.

### Isolation Forest

An unsupervised model identifies unusual combinations of operational metrics.

---

# Key Results From the Supplied Dataset

The executed pipeline found:

| Metric | Result |
|---|---:|
| Records | 1,890 |
| Unique vessels | 342 |
| Privileged | 676 |
| Taxed | 582 |
| Blocked | 632 |
| Asset-value formula mismatches | 0 |
| Transit-cost formula mismatches | 0 |
| Cost z-score anomalies | 3 |
| Isolation Forest anomalies | 5 |

The Welch test comparing simulated Taxed vs Privileged transit costs returned:

```text
p ≈ 6.46 × 10^-277
```

The chi-square test for trade tier vs rerouting returned:

```text
p ≈ 5.78 × 10^-212
```

XGBoost produced a test-set:

```text
ROC-AUC = 1.00
F1       = 1.00
Accuracy = 1.00
```

### How to interpret the perfect ML result

This should **not** be presented as evidence that real-world vessel rerouting can be predicted with 100% accuracy.

The dataset is simulated and appears to contain highly deterministic relationships between its variables and the rerouting outcome. This is itself an important modeling observation and should be investigated rather than hidden.

---


# Business Interpretation Framework

The project does not attempt to determine whether any political actor or geopolitical strategy is "good" or "bad."

Instead, it quantifies simulated operational consequences:

```text
Trade Tier
     ↓
Cost / Delay / Routing
     ↓
Financial Exposure
     ↓
Supply Chain Risk
```

This keeps the analysis focused on measurable data rather than political conclusions.

---

# Limitations

1. The dataset is simulated rather than independently verified historical AIS/trade data.
2. Results are dependent on the assumptions used to generate the simulation.
3. Statistical significance does not establish causality.
4. Perfect classification performance may indicate deterministic synthetic relationships.
5. Missing values exist in some variables and must be handled carefully.
6. The dataset should not be used to make real-world geopolitical forecasts.
7. The project measures relationships within the supplied data rather than predicting actual future maritime behavior.

