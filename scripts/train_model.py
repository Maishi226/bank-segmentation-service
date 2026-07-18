"""Generate synthetic bank data, train K-Means, and export labeled customers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 20260711
CUSTOMER_COUNT = 1000

MODEL_FEATURES = [
    "avg_monthly_inflow_6m", "salary_inflow_ratio", "inflow_cv_6m",
    "avg_balance_6m", "min_balance_6m", "avg_monthly_spend_6m",
    "monthly_txn_count_6m", "digital_txn_ratio", "cash_withdrawal_ratio",
    "discretionary_spend_ratio", "travel_spend_ratio",
    "investment_contribution_ratio", "credit_card_utilisation",
    "days_since_last_txn", "monthly_app_logins_3m", "products_held",
    "overdraft_events_6m",
]

LOG_FEATURES = [
    "avg_monthly_inflow_6m", "avg_balance_6m", "min_balance_6m",
    "avg_monthly_spend_6m", "monthly_txn_count_6m",
    "days_since_last_txn", "monthly_app_logins_3m", "overdraft_events_6m",
]

LINEAR_FEATURES = [feature for feature in MODEL_FEATURES if feature not in LOG_FEATURES]

ARCHETYPES = [
    {"weight": .23, "age": 25, "tenure": 30, "inflow": 3300, "salary": .72, "cv": .24, "balance": 3100, "min_ratio": .28, "spend": .78, "tx": 72, "digital": .92, "cash": .05, "discretionary": .36, "travel": .04, "education": .09, "business": .04, "investment": .03, "loan": 8500, "cc": .34, "recency": 2, "logins": 24, "products": 2, "overdraft": .35, "late": .08},
    {"weight": .26, "age": 39, "tenure": 92, "inflow": 6200, "salary": .86, "cv": .12, "balance": 14500, "min_ratio": .48, "spend": .58, "tx": 61, "digital": .82, "cash": .08, "discretionary": .27, "travel": .05, "education": .07, "business": .05, "investment": .08, "loan": 145000, "cc": .25, "recency": 2, "logins": 18, "products": 4, "overdraft": .12, "late": .03},
    {"weight": .15, "age": 51, "tenure": 150, "inflow": 16500, "salary": .58, "cv": .18, "balance": 118000, "min_ratio": .68, "spend": .35, "tx": 48, "digital": .76, "cash": .04, "discretionary": .30, "travel": .13, "education": .03, "business": .12, "investment": .28, "loan": 210000, "cc": .17, "recency": 3, "logins": 14, "products": 6, "overdraft": .03, "late": .01},
    {"weight": .08, "age": 35, "tenure": 68, "inflow": 9000, "salary": .63, "cv": .24, "balance": 8500, "min_ratio": .15, "spend": .96, "tx": 102, "digital": .90, "cash": .05, "discretionary": .50, "travel": .13, "education": .03, "business": .08, "investment": .03, "loan": 74000, "cc": .70, "recency": 1, "logins": 29, "products": 4, "overdraft": 2.3, "late": .20},
    {"weight": .12, "age": 37, "tenure": 79, "inflow": 7800, "salary": .69, "cv": .20, "balance": 10500, "min_ratio": .23, "spend": .83, "tx": 88, "digital": .87, "cash": .07, "discretionary": .45, "travel": .10, "education": .04, "business": .07, "investment": .05, "loan": 90000, "cc": .62, "recency": 1, "logins": 25, "products": 4, "overdraft": .35, "late": .12},
    {"weight": .16, "age": 57, "tenure": 125, "inflow": 4100, "salary": .43, "cv": .30, "balance": 10600, "min_ratio": .42, "spend": .54, "tx": 23, "digital": .28, "cash": .39, "discretionary": .19, "travel": .02, "education": .02, "business": .09, "investment": .05, "loan": 27000, "cc": .12, "recency": 14, "logins": 3, "products": 2, "overdraft": .25, "late": .06},
]


def clipped_normal(rng, mean, spread, low, high):
    return float(np.clip(rng.normal(mean, spread), low, high))


def generate_customers(count: int = CUSTOMER_COUNT) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    weights = np.array([profile["weight"] for profile in ARCHETYPES])
    weights = weights / weights.sum()
    records = []
    regions = ["Auckland", "Wellington", "Canterbury", "Waikato", "Bay of Plenty", "Other"]

    for index in range(count):
        profile = ARCHETYPES[int(rng.choice(len(ARCHETYPES), p=weights))]
        age = int(np.clip(round(rng.normal(profile["age"], 6)), 18, 82))
        tenure = int(np.clip(round(rng.normal(profile["tenure"], 25)), 1, max(1, age * 12 - 210)))
        inflow = max(650, profile["inflow"] * np.exp(rng.normal(0, .23)))
        balance = max(50, profile["balance"] * np.exp(rng.normal(0, .42)))
        spend = max(250, inflow * clipped_normal(rng, profile["spend"], .09, .18, 1.25))
        discretionary = clipped_normal(rng, profile["discretionary"], .05, .08, .65)
        essential = clipped_normal(rng, .58 - discretionary * .35, .045, .25, .78)

        records.append({
            "customer_id": f"C{index + 1:05d}",
            "snapshot_date": "2026-06-30",
            "age": age,
            "region": str(rng.choice(regions)),
            "relationship_months": tenure,
            "avg_monthly_inflow_6m": round(inflow),
            "salary_inflow_ratio": round(clipped_normal(rng, profile["salary"], .09, 0, 1), 3),
            "inflow_cv_6m": round(clipped_normal(rng, profile["cv"], .05, .02, .85), 3),
            "avg_balance_6m": round(balance),
            "min_balance_6m": round(max(0, balance * clipped_normal(rng, profile["min_ratio"], .09, .02, .9))),
            "avg_monthly_spend_6m": round(spend),
            "monthly_txn_count_6m": int(np.clip(round(rng.normal(profile["tx"], 11)), 3, 160)),
            "digital_txn_ratio": round(clipped_normal(rng, profile["digital"], .07, 0, 1), 3),
            "cash_withdrawal_ratio": round(clipped_normal(rng, profile["cash"], .05, 0, .75), 3),
            "essential_spend_ratio": round(essential, 3),
            "discretionary_spend_ratio": round(discretionary, 3),
            "travel_spend_ratio": round(clipped_normal(rng, profile["travel"], .023, 0, .3), 3),
            "education_spend_ratio": round(clipped_normal(rng, profile["education"], .02, 0, .25), 3),
            "business_merchant_ratio": round(clipped_normal(rng, profile["business"], .03, 0, .35), 3),
            "investment_contribution_ratio": round(clipped_normal(rng, profile["investment"], .035, 0, .45), 3),
            "loan_balance": round(max(0, profile["loan"] * (0 if rng.random() < .18 else np.exp(rng.normal(0, .55))))),
            "credit_card_utilisation": round(clipped_normal(rng, profile["cc"], .10, 0, .98), 3),
            "days_since_last_txn": int(np.clip(round(profile["recency"] + abs(rng.normal()) * profile["recency"]), 0, 90)),
            "monthly_app_logins_3m": int(np.clip(round(rng.normal(profile["logins"], 5)), 0, 55)),
            "products_held": int(np.clip(round(rng.normal(profile["products"], .8)), 1, 8)),
            "overdraft_events_6m": int(rng.poisson(profile["overdraft"])),
            "late_payment_events_12m": int(rng.poisson(profile["late"])),
        })
    return pd.DataFrame(records)


def create_preprocessor() -> ColumnTransformer:
    log_pipeline = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    linear_pipeline = Pipeline([("scale", StandardScaler())])
    return ColumnTransformer([
        ("log", log_pipeline, LOG_FEATURES),
        ("linear", linear_pipeline, LINEAR_FEATURES),
    ])


def assign_segment_names(frame: pd.DataFrame) -> dict[int, str]:
    profile = frame.groupby("model_cluster").agg(
        digital=("digital_txn_ratio", "mean"),
        cash=("cash_withdrawal_ratio", "mean"),
        balance=("avg_balance_6m", "mean"),
        investment=("investment_contribution_ratio", "mean"),
        salary=("salary_inflow_ratio", "mean"),
        cc=("credit_card_utilisation", "mean"),
        overdraft=("overdraft_events_6m", "mean"),
        spend=("avg_monthly_spend_6m", "mean"),
        inflow=("avg_monthly_inflow_6m", "mean"),
    )
    remaining = set(int(value) for value in profile.index)
    mapping = {}

    def take(cluster, label):
        mapping[int(cluster)] = label
        remaining.remove(int(cluster))

    take(profile.loc[list(remaining), "digital"].idxmin(), "Low Digital Engagement and Cash-Oriented")
    take(profile.loc[list(remaining), "investment"].idxmax(), "Affluent Investors")
    salary_score = profile["salary"] + profile["balance"].rank(pct=True) * .2
    take(salary_score.loc[list(remaining)].idxmax(), "Stable Salary Builders")
    take(profile.loc[list(remaining), "overdraft"].idxmax(), "High Spend and Frequent Overdraft")
    take(profile.loc[list(remaining), "cc"].idxmax(), "High Spend Credit Active")
    for cluster in list(remaining):
        take(cluster, "Emerging Digital Everyday")
    return mapping


def main() -> None:
    raw = generate_customers()
    raw_path = ROOT / "data" / "raw" / "synthetic_bank_customers.csv"
    labeled_path = ROOT / "data" / "processed" / "labeled_bank_customers.csv"
    artifact_path = ROOT / "artifacts" / "segmentation_model.joblib"
    metadata_path = ROOT / "artifacts" / "model_metadata.json"
    for path in [raw_path, labeled_path, artifact_path, metadata_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)

    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(raw[MODEL_FEATURES])
    evaluations = []
    candidates = {}
    for k in range(3, 9):
        model = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=30)
        labels = model.fit_predict(transformed)
        score = float(silhouette_score(transformed, labels))
        evaluations.append({"k": k, "inertia": float(model.inertia_), "silhouette_score": score})
        candidates[k] = model

    # The demo requires six actionable advertising audiences. Candidate scores
    # are retained for review, while K=6 is selected as the business constraint.
    selected_k = 6
    model = candidates[selected_k]
    raw["model_cluster"] = model.labels_
    name_mapping = assign_segment_names(raw)

    # Stable public IDs are assigned alphabetically by business label.
    ordered_labels = sorted(name_mapping.values())
    public_id_by_name = {name: index + 1 for index, name in enumerate(ordered_labels)}
    raw["segment_name"] = raw["model_cluster"].map(name_mapping)
    raw["segment_id"] = raw["segment_name"].map(public_id_by_name)
    distances = model.transform(transformed)
    assigned_distance = distances[np.arange(len(raw)), model.labels_]
    distance_scale = {
        int(cluster): float(np.mean(assigned_distance[model.labels_ == cluster]))
        for cluster in range(selected_k)
    }
    raw["assignment_confidence"] = [
        float(np.clip(np.exp(-distance / (distance_scale[int(cluster)] * 1.7)), .05, .99))
        for distance, cluster in zip(assigned_distance, model.labels_)
    ]
    raw["distance_to_center"] = assigned_distance
    raw = raw.drop(columns=["model_cluster"])
    output_columns = ["customer_id", "segment_id", "segment_name", "assignment_confidence", "distance_to_center"] + [
        column for column in raw.columns if column not in {"customer_id", "segment_id", "segment_name", "assignment_confidence", "distance_to_center"}
    ]
    raw[output_columns].to_csv(labeled_path, index=False)

    bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "model_features": MODEL_FEATURES,
        "cluster_to_segment_name": name_mapping,
        "segment_name_to_id": public_id_by_name,
        "distance_scale": distance_scale,
        "model_version": "1.0.0-demo",
        "snapshot_date": "2026-06-30",
    }
    joblib.dump(bundle, artifact_path)
    metadata = {
        "model_version": bundle["model_version"],
        "random_seed": RANDOM_SEED,
        "customer_count": len(raw),
        "selected_k": selected_k,
        "model_features": MODEL_FEATURES,
        "excluded_from_model": ["customer_id", "snapshot_date", "age", "region", "relationship_months", "essential_spend_ratio", "education_spend_ratio", "business_merchant_ratio", "loan_balance", "late_payment_events_12m"],
        "candidate_evaluations": evaluations,
        "segments": [
            {"segment_id": public_id_by_name[name], "segment_name": name}
            for name in ordered_labels
        ],
        "limitations": [
            "The data is synthetic and is not representative of any real bank population.",
            "Assignment confidence is a distance-based demo score, not a calibrated probability.",
            "The model does not infer occupation, wealth status, student status, or purchase intent.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"raw": str(raw_path), "labeled": str(labeled_path), "model": str(artifact_path), "selected_k": selected_k}, indent=2))


if __name__ == "__main__":
    main()
