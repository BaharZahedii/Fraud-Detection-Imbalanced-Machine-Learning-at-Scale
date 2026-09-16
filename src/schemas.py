from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    """A single raw transaction, in the same shape as one row of the
    PaySim dataset (minus the label)."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step": 743,
                "type": "TRANSFER",
                "amount": 181000.00,
                "nameOrig": "C1231006815",
                "oldbalanceOrg": 181000.00,
                "newbalanceOrig": 0.00,
                "nameDest": "C1900366749",
                "oldbalanceDest": 0.00,
                "newbalanceDest": 0.00,
            }
        }
    )

    step: int = Field(..., description="Time unit, roughly 1 hour per step")
    type: str = Field(..., description="PAYMENT, CASH_OUT, CASH_IN, TRANSFER, DEBIT")
    amount: float
    nameOrig: str = Field(..., description="Sender account ID")
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str = Field(..., description="Recipient account ID")
    oldbalanceDest: float
    newbalanceDest: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud_prediction: bool
    decision_threshold: float
    model_version: str
    sender_transaction_history_count: int
