"""Integration checks for model artifacts and API-facing service behavior."""

from app.service import SegmentationService


def main() -> None:
    service = SegmentationService()
    assert len(service.customers) == 1000
    assert len(service.segments()) >= 3
    customer = service.customer("C00001")
    assert customer and customer["segment_name"]
    page = service.customers_for_segment(int(customer["segment_id"]), 10, 0)
    assert page["customers"] and all(row["segment_id"] == customer["segment_id"] for row in page["customers"])
    record = {feature: customer[feature] for feature in service.model_features}
    record["customer_id"] = customer["customer_id"]
    predicted = service.segment_records([record])[0]
    assert predicted["segment_id"] == customer["segment_id"]
    print("All service checks passed.")


if __name__ == "__main__":
    main()
