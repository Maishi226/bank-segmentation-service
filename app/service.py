"""Model loading and customer segmentation business logic."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class SegmentationService:
    def __init__(self) -> None:
        self.bundle = joblib.load(ROOT / "artifacts" / "segmentation_model.joblib")
        self.customers = pd.read_csv(ROOT / "data" / "processed" / "labeled_bank_customers.csv")

    @property
    def model_features(self) -> list[str]:
        return self.bundle["model_features"]

    def segments(self) -> list[dict]:
        grouped = self.customers.groupby(["segment_id", "segment_name"]).agg(
            customer_count=("customer_id", "count"),
            average_confidence=("assignment_confidence", "mean"),
            average_balance_nzd=("avg_balance_6m", "mean"),
            average_monthly_inflow_nzd=("avg_monthly_inflow_6m", "mean"),
        ).reset_index()
        return grouped.round(4).to_dict(orient="records")

    def customer(self, customer_id: str) -> dict | None:
        match = self.customers[self.customers["customer_id"] == customer_id]
        return None if match.empty else match.to_dict(orient="records")[0]

    def customers_for_segment(self, segment_id: int | None, limit: int, offset: int) -> dict:
        selected = self.customers
        if segment_id is not None:
            selected = selected[selected["segment_id"] == segment_id]
        total = len(selected)
        columns = ["customer_id", "segment_id", "segment_name", "assignment_confidence"]
        records = selected.iloc[offset:offset + limit][columns].round(4).to_dict(orient="records")
        return {"total": total, "offset": offset, "limit": limit, "customers": records}

    def segment_records(self, records: list[dict]) -> list[dict]:
        missing = [
            {"record_index": index, "missing_fields": sorted(set(self.model_features) - set(record))}
            for index, record in enumerate(records)
            if set(self.model_features) - set(record)
        ]
        if missing:
            raise ValueError(f"Missing required model fields: {missing}")

        frame = pd.DataFrame(records)
        transformed = self.bundle["preprocessor"].transform(frame[self.model_features])
        model_clusters = self.bundle["model"].predict(transformed)
        distances = self.bundle["model"].transform(transformed)
        output = []
        for index, cluster in enumerate(model_clusters):
            cluster = int(cluster)
            name = self.bundle["cluster_to_segment_name"][cluster]
            public_id = int(self.bundle["segment_name_to_id"][name])
            distance = float(distances[index, cluster])
            scale = float(self.bundle["distance_scale"][cluster])
            confidence = float(np.clip(np.exp(-distance / (scale * 1.7)), .05, .99))
            output.append({
                "customer_id": records[index].get("customer_id"),
                "segment_id": public_id,
                "segment_name": name,
                "assignment_confidence": round(confidence, 4),
                "model_version": self.bundle["model_version"],
            })
        return output
