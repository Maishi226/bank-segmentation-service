"""FastAPI application for the bank customer segmentation service."""

from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from app.service import SegmentationService


SERVICE = SegmentationService()

app = FastAPI(
    title="Bank Customer Segmentation API",
    description=(
        "Returns existing customer segment labels and assigns segment labels "
        "to new bank-owned feature records. This service does not deliver ads."
    ),
    version=SERVICE.bundle["model_version"],
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class CustomerFeatures(BaseModel):
    """Bank-owned fields required for real-time segmentation."""

    model_config = ConfigDict(extra="ignore")

    customer_id: str | None = Field(default=None, examples=["NEW0001"])
    avg_monthly_inflow_6m: float = Field(ge=0)
    salary_inflow_ratio: float = Field(ge=0, le=1)
    inflow_cv_6m: float = Field(ge=0)
    avg_balance_6m: float = Field(ge=0)
    min_balance_6m: float = Field(ge=0)
    avg_monthly_spend_6m: float = Field(ge=0)
    monthly_txn_count_6m: float = Field(ge=0)
    digital_txn_ratio: float = Field(ge=0, le=1)
    cash_withdrawal_ratio: float = Field(ge=0, le=1)
    discretionary_spend_ratio: float = Field(ge=0, le=1)
    travel_spend_ratio: float = Field(ge=0, le=1)
    investment_contribution_ratio: float = Field(ge=0, le=1)
    credit_card_utilisation: float = Field(ge=0, le=1)
    days_since_last_txn: float = Field(ge=0)
    monthly_app_logins_3m: float = Field(ge=0)
    products_held: float = Field(ge=0)
    overdraft_events_6m: float = Field(ge=0)


class SegmentRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=500)


class SegmentResult(BaseModel):
    customer_id: str | None
    segment_id: int
    segment_name: str
    assignment_confidence: float
    model_version: str


class SegmentResponse(BaseModel):
    results: list[SegmentResult]


@app.get("/health", tags=["Operations"])
def health() -> dict:
    """Return service and model health."""
    return {"status": "ok", "model_version": SERVICE.bundle["model_version"]}


@app.get("/v1/segments", tags=["Segments"])
def list_segments() -> dict:
    """Return available segment metadata and customer counts."""
    return {"segments": SERVICE.segments()}


@app.get("/v1/customers", tags=["Customers"])
def list_customers(
    segment_id: int | None = Query(default=None, ge=1),
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """Return already-labeled customers, optionally filtered by segment."""
    return SERVICE.customers_for_segment(segment_id, limit, offset)


@app.get("/v1/customers/{customer_id}", tags=["Customers"])
def get_customer(customer_id: str) -> dict:
    """Return one labeled customer record."""
    customer = SERVICE.customer(customer_id)
    if customer is None:
        raise HTTPException(status_code=404, detail="Customer not found")
    return customer


@app.get("/v1/model/features", tags=["Model"])
def model_features() -> dict:
    """Return the exact bank feature contract required for scoring."""
    return {"required_features": SERVICE.model_features}


@app.post("/v1/segment", response_model=SegmentResponse, tags=["Model"])
def segment_customers(request: SegmentRequest) -> SegmentResponse:
    """Assign segment labels to up to 500 customer feature records."""
    try:
        records = [customer.model_dump() for customer in request.customers]
        return SegmentResponse(results=SERVICE.segment_records(records))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
