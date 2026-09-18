
# Strait of Hormuz 2026 — Maritime Trade & Risk Analytics
# Run from the project root:
#     python hormuz_trade_analytics.py

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "hormuz_trade_tier_continental_2026.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


# -----------------------------
# 1. Load + data quality
# -----------------------------
df = pd.read_csv(DATA)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["reroute_flag"] = (df["rerouted"] == "Yes (Cape of Good Hope)").astype(int)

quality = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing": [df[c].isna().sum() for c in df.columns],
    "missing_pct": [df[c].isna().mean() * 100 for c in df.columns],
    "unique": [df[c].nunique(dropna=True) for c in df.columns],
})
quality.to_csv(OUT / "data_quality_report.csv", index=False)

# Check arithmetic consistency
df["asset_value_check_diff"] = (
    df["total_asset_value_at_risk_usd"]
    - df["ship_hull_value_usd"]
    - df["estimated_cargo_value_usd"]
)

df["transit_cost_check_diff"] = (
    df["total_transit_cost_usd"]
    - df["toll_usd"]
    - df["insurance_cost_usd"]
    - df["reroute_penalty_usd"]
)

consistency = pd.DataFrame({
    "check": [
        "asset_value = hull + cargo",
        "transit_cost = toll + insurance + reroute_penalty",
    ],
    "rows_with_difference": [
        (df["asset_value_check_diff"] != 0).sum(),
        (df["transit_cost_check_diff"] != 0).sum(),
    ],
})
consistency.to_csv(OUT / "consistency_checks.csv", index=False)


# -----------------------------
# 2. Descriptive analytics
# -----------------------------
tier_summary = (
    df.groupby("trade_tier")
      .agg(
          records=("mmsi", "size"),
          unique_vessels=("mmsi", "nunique"),
          reroute_rate=("reroute_flag", "mean"),
          avg_delay_days=("days_delayed", "mean"),
          avg_transit_cost_usd=("total_transit_cost_usd", "mean"),
          total_transit_cost_usd=("total_transit_cost_usd", "sum"),
          total_asset_risk_usd=("total_asset_value_at_risk_usd", "sum"),
          avg_insurance_cost_usd=("insurance_cost_usd", "mean"),
          avg_extra_fuel_tonnes=("extra_fuel_tonnes", "mean"),
      )
      .reset_index()
)
tier_summary["reroute_rate_pct"] = tier_summary["reroute_rate"] * 100
tier_summary.to_csv(OUT / "trade_tier_summary.csv", index=False)

payment_summary = (
    df.groupby("payment_rail")
      .agg(
          records=("mmsi", "size"),
          reroute_rate=("reroute_flag", "mean"),
          avg_transit_cost_usd=("total_transit_cost_usd", "mean"),
          avg_delay_days=("days_delayed", "mean"),
          total_asset_risk_usd=("total_asset_value_at_risk_usd", "sum"),
      )
      .reset_index()
)
payment_summary["reroute_rate_pct"] = payment_summary["reroute_rate"] * 100
payment_summary.to_csv(OUT / "payment_rail_summary.csv", index=False)

continent_summary = (
    df.groupby("continent")
      .agg(
          records=("mmsi", "size"),
          reroute_rate=("reroute_flag", "mean"),
          avg_transit_cost_usd=("total_transit_cost_usd", "mean"),
          avg_delay_days=("days_delayed", "mean"),
          total_asset_risk_usd=("total_asset_value_at_risk_usd", "sum"),
      )
      .reset_index()
)
continent_summary["reroute_rate_pct"] = continent_summary["reroute_rate"] * 100
continent_summary.to_csv(OUT / "continent_summary.csv", index=False)


# -----------------------------
# 3. Statistical tests
# -----------------------------
taxed = df.loc[df["trade_tier"] == "Taxed", "total_transit_cost_usd"]
priv = df.loc[df["trade_tier"] == "Privileged", "total_transit_cost_usd"]

tt = ttest_ind(taxed, priv, equal_var=False)
mw = mannwhitneyu(taxed, priv, alternative="two-sided")

contingency = pd.crosstab(df["trade_tier"], df["reroute_flag"])
chi2, chi_p, chi_dof, chi_expected = chi2_contingency(contingency)

stats = pd.DataFrame({
    "test": [
        "Welch t-test: Taxed vs Privileged transit cost",
        "Mann-Whitney U: Taxed vs Privileged transit cost",
        "Chi-square: Trade tier vs rerouting",
    ],
    "statistic": [tt.statistic, mw.statistic, chi2],
    "p_value": [tt.pvalue, mw.pvalue, chi_p],
    "sample_note": [
        "Two-sided; unequal variances",
        "Two-sided non-parametric test",
        "Independence test",
    ],
})
stats.to_csv(OUT / "statistical_tests.csv", index=False)


# -----------------------------
# 4. Daily operational analytics
# -----------------------------
daily = (
    df.groupby("date")
      .agg(
          vessel_records=("mmsi", "size"),
          unique_vessels=("mmsi", "nunique"),
          reroute_rate=("reroute_flag", "mean"),
          avg_delay_days=("days_delayed", "mean"),
          total_transit_cost_usd=("total_transit_cost_usd", "sum"),
          total_asset_risk_usd=("total_asset_value_at_risk_usd", "sum"),
          total_fuel_tonnes=("extra_fuel_tonnes", "sum"),
      )
      .reset_index()
      .sort_values("date")
)

daily["reroute_rate_pct"] = daily["reroute_rate"] * 100
daily["rolling_7d_cost_usd"] = daily["total_transit_cost_usd"].rolling(
    7, min_periods=3
).mean()

rolling_mean = daily["total_transit_cost_usd"].rolling(30, min_periods=10).mean()
rolling_std = daily["total_transit_cost_usd"].rolling(30, min_periods=10).std()
daily["cost_z_30d"] = (
    daily["total_transit_cost_usd"] - rolling_mean
) / rolling_std
daily["cost_anomaly"] = daily["cost_z_30d"].abs() > 3

daily.to_csv(OUT / "daily_operational_metrics.csv", index=False)


# -----------------------------
# 5. Classification — rerouting
# IMPORTANT: transit_status and outcome-derived
# cost/delay variables are deliberately excluded
# to avoid target leakage.
# -----------------------------
model_features = [
    "trade_tier",
    "flag",
    "commodity",
    "destination",
    "payment_rail",
    "continent",
    "ship_hull_value_usd",
    "estimated_cargo_value_usd",
    "insurance_premium_delta_pct",
]

X = df[model_features].copy()
y = df["reroute_flag"].copy()

# Chronological holdout: first 70% of dates for training, remaining 30% for test.
dates = sorted(df["date"].dropna().unique())
cutoff = dates[int(len(dates) * 0.70)]
train_mask = df["date"] < cutoff
test_mask = ~train_mask

X_train, X_test = X.loc[train_mask], X.loc[test_mask]
y_train, y_test = y.loc[train_mask], y.loc[test_mask]

categorical = [c for c in model_features if X[c].dtype == "object"]
numeric = [c for c in model_features if c not in categorical]

preprocessor = ColumnTransformer([
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical,
    ),
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric,
    ),
])

lr = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
lr.fit(X_train, y_train)
lr_prob = lr.predict_proba(X_test)[:, 1]
lr_pred = (lr_prob >= 0.5).astype(int)

# Fit preprocessing separately for XGBoost so we can inspect feature names.
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

xgb = XGBClassifier(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42,
    n_jobs=2,
)
xgb.fit(X_train_enc, y_train)
xgb_prob = xgb.predict_proba(X_test_enc)[:, 1]
xgb_pred = (xgb_prob >= 0.5).astype(int)

def metrics_row(name, pred, prob):
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob),
    }

model_results = pd.DataFrame([
    metrics_row("Logistic Regression", lr_pred, lr_prob),
    metrics_row("XGBoost", xgb_pred, xgb_prob),
])
model_results.to_csv(OUT / "model_comparison.csv", index=False)

pd.DataFrame(
    confusion_matrix(y_test, xgb_pred),
    index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"],
).to_csv(OUT / "xgb_confusion_matrix.csv")

# Feature importance
feature_names = preprocessor.get_feature_names_out()
importance = pd.DataFrame({
    "feature": feature_names,
    "importance": xgb.feature_importances_,
}).sort_values("importance", ascending=False)
importance.to_csv(OUT / "xgb_feature_importance.csv", index=False)

# Vessel-level test predictions
predictions = df.loc[test_mask, [
    "date", "vessel_name", "mmsi", "flag", "trade_tier",
    "commodity", "destination", "payment_rail", "rerouted"
]].copy()
predictions["actual_reroute"] = y_test.values
predictions["xgb_reroute_probability"] = xgb_prob
predictions["xgb_prediction"] = xgb_pred
predictions.to_csv(OUT / "reroute_predictions_test_set.csv", index=False)


# -----------------------------
# 6. Anomaly detection
# -----------------------------
anomaly_features = [
    "vessel_records",
    "reroute_rate_pct",
    "avg_delay_days",
    "total_transit_cost_usd",
    "total_asset_risk_usd",
    "total_fuel_tonnes",
]

A = daily[anomaly_features].replace([np.inf, -np.inf], np.nan).fillna(0)

iso = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42,
)
daily["isolation_forest_label"] = iso.fit_predict(A)
daily["isolation_forest_anomaly"] = daily["isolation_forest_label"].eq(-1)

daily.to_csv(OUT / "daily_operational_metrics_with_anomalies.csv", index=False)


# -----------------------------
# 7. Charts
# -----------------------------
plt.figure(figsize=(11, 5))
plt.plot(daily["date"], daily["total_transit_cost_usd"], label="Daily total transit cost")
plt.plot(daily["date"], daily["rolling_7d_cost_usd"], label="7-day rolling average")
plt.title("Daily Transit Cost Through the Simulated Hormuz Crisis")
plt.xlabel("Date")
plt.ylabel("USD")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "01_daily_transit_cost.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 5))
tier_plot = tier_summary.sort_values("avg_transit_cost_usd")
plt.bar(tier_plot["trade_tier"], tier_plot["avg_transit_cost_usd"])
plt.title("Average Transit Cost by Trade Tier")
plt.xlabel("Trade Tier")
plt.ylabel("Average Transit Cost (USD)")
plt.tight_layout()
plt.savefig(OUT / "02_avg_cost_by_trade_tier.png", dpi=180)
plt.close()

plt.figure(figsize=(9, 5))
plt.bar(continent_summary["continent"], continent_summary["reroute_rate_pct"])
plt.title("Rerouting Rate by Continent")
plt.xlabel("Continent")
plt.ylabel("Rerouting Rate (%)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT / "03_reroute_rate_by_continent.png", dpi=180)
plt.close()

plt.figure(figsize=(11, 5))
plt.plot(daily["date"], daily["reroute_rate_pct"])
plt.title("Daily Rerouting Rate")
plt.xlabel("Date")
plt.ylabel("Rerouting Rate (%)")
plt.tight_layout()
plt.savefig(OUT / "04_daily_reroute_rate.png", dpi=180)
plt.close()

top_features = importance.head(12).sort_values("importance")
plt.figure(figsize=(9, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.title("XGBoost Feature Importance for Rerouting Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(OUT / "05_xgb_feature_importance.png", dpi=180)
plt.close()


# -----------------------------
# 8. Project summary
# -----------------------------
summary = {
    "records": int(len(df)),
    "date_start": str(df["date"].min().date()),
    "date_end": str(df["date"].max().date()),
    "unique_vessels": int(df["mmsi"].nunique()),
    "trade_tiers": df["trade_tier"].value_counts().to_dict(),
    "missing_naval_escort_rows": int(df["naval_escort_status"].isna().sum()),
    "missing_inflation_premium_rows": int(df["inflation_premium_per_unit"].isna().sum()),
    "asset_formula_mismatches": int((df["asset_value_check_diff"] != 0).sum()),
    "transit_cost_formula_mismatches": int((df["transit_cost_check_diff"] != 0).sum()),
    "welch_ttest_pvalue_taxed_vs_privileged": float(tt.pvalue),
    "chi_square_pvalue_trade_tier_vs_reroute": float(chi_p),
    "xgb_test_roc_auc": float(model_results.loc[model_results.model == "XGBoost", "roc_auc"].iloc[0]),
    "xgb_test_f1": float(model_results.loc[model_results.model == "XGBoost", "f1"].iloc[0]),
    "xgb_test_accuracy": float(model_results.loc[model_results.model == "XGBoost", "accuracy"].iloc[0]),
    "cost_z_anomalies": int(daily["cost_anomaly"].sum()),
    "isolation_forest_anomalies": int(daily["isolation_forest_anomaly"].sum()),
}
with open(OUT / "project_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nPROJECT COMPLETE")
print(json.dumps(summary, indent=2))
print("\nOutputs written to:", OUT)
