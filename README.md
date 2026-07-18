# Bank Customer Segmentation Service
## Companion Demo

This service is used by the BNZ AI marketing hybrid demo:

https://github.com/Maishi226/bnz-ai-marketing-hybrid

This demo turns synthetic bank-owned customer behavior data into labeled customer records and exposes those labels through a FastAPI REST service. No survey, uploaded form, name, phone number, or identity document is required.

## Why FastAPI

FastAPI converts typed Python functions into validated HTTP endpoints. It automatically provides:

- Interactive Swagger documentation at `http://localhost:8000/docs`.
- An OpenAPI contract at `http://localhost:8000/openapi.json`.
- Request validation through Pydantic models.
- Clear HTTP error responses when fields are missing or invalid.
- Easy integration with Python, JavaScript, Java, mobile, or campaign systems.

## End-to-end flow

1. `scripts/train_model.py` generates 1,000 synthetic customer records using fields normally available in core banking, transaction, card, channel, product, and lending systems.
2. The training pipeline applies `log1p` to skewed monetary/count features and standardizes all model features.
3. K-Means candidates with K=3 through K=8 are compared using Silhouette Score.
4. The selected model produces `segment_id`, `segment_name`, `assignment_confidence`, and `distance_to_center`.
5. `data/processed/labeled_bank_customers.csv` is the labeled dataset.
6. The API lets another service retrieve already-labeled customers or label new customer feature records.

## Files

- `data/raw/synthetic_bank_customers.csv`: synthetic bank-owned input data without labels.
- `data/processed/labeled_bank_customers.csv`: model output with customer labels.
- `artifacts/segmentation_model.joblib`: fitted preprocessing and K-Means model bundle.
- `artifacts/model_metadata.json`: model features, K comparison, labels, and limitations.
- `Bank_Customer_Segmentation_Demo_EN.xlsx`: English stakeholder workbook.

## Run locally

### One-command setup on macOS or Linux

```bash
chmod +x setup_and_run.sh run_tests.sh
./setup_and_run.sh
```

Open a second terminal and run:

```bash
./run_tests.sh
```

### Manual setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
python -m app.main
```

The API listens on `http://localhost:8000`.

After starting it, open `http://localhost:8000/docs` in a browser. Every endpoint can be tested from the Swagger page without writing additional code.

## API endpoints

Interactive API documentation: `http://localhost:8000/docs`

### Health check

```bash
curl http://localhost:8000/health
```

### List available segments

```bash
curl http://localhost:8000/v1/segments
```

### Get customers already assigned to a segment

This is the primary endpoint for the advertising system.

```bash
curl "http://localhost:8000/v1/customers?segment_id=3&limit=100"
```

Response:

```json
{
  "total": 152,
  "offset": 0,
  "limit": 100,
  "customers": [
    {
      "customer_id": "C00017",
      "segment_id": 3,
      "segment_name": "High Spend Credit Active",
      "assignment_confidence": 0.61
    }
  ]
}
```

### Get one labeled customer

```bash
curl http://localhost:8000/v1/customers/C00017
```

### Label new bank feature records

First retrieve the exact feature contract:

```bash
curl http://localhost:8000/v1/model/features
```

Then submit up to 500 records:

```bash
curl -X POST http://localhost:8000/v1/segment \
  -H "Content-Type: application/json" \
  --data @docs/example_segment_request.json
```

## Advertising integration

The ad system should keep creative selection separate from the segmentation model. A recommended integration is:

1. Call `GET /v1/customers?segment_id={id}` to retrieve eligible customer IDs.
2. Intersect the result with marketing-consent and channel-eligibility records.
3. Select a creative mapped to the returned segment.
4. Apply frequency caps, exclusions, and campaign rules.
5. Log exposure and response for later A/B testing.

The segmentation API must not directly send an advertisement. It supplies an audience label; the campaign system remains responsible for consent, eligibility, creative selection, and delivery.

## Important limitations

- All customers and values are synthetic.
- The demo confidence value is distance-based and is not a calibrated probability.
- Clustering does not prove occupation, wealth, student status, or purchase intent.
- Before production use, review privacy, fairness, marketing consent, security, access control, audit logging, model drift, and data retention requirements.
