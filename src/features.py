"""
Leakage-safe feature engineering, shared by training and serving.

Two code paths compute the SAME features, because they must stay consistent
with what the model was trained on:

1. `engineer_features_batch(df)`
   Used at TRAINING time. Vectorised, uses a full historical dataframe
   (via groupby/cumcount/shift), exactly like the original notebook.

2. `SenderHistoryStore`
   Used at SERVING time. A single transaction arrives with no access to the
   full dataset, so historical aggregates (previous transaction count,
   time since last transaction, cumulative amount, etc.) have to be
   maintained incrementally, per sender, in a small stateful store.

In this reference implementation the store is an in-memory Python dict.
That is fine for a demo / single-process API, but it does NOT persist
across restarts and does NOT work across multiple API replicas. For a
real deployment this class would be backed by a fast key-value store
(e.g. Redis) or a feature-store product, keyed by `nameOrig`. See the
README for more detail on this trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

# The exact feature set the model is trained and served on.
FEATURE_COLUMNS = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "hist_per_sender", "time_since_last",
    "AVG_txn_amount", "total_past_amount",
    "balance_change", "insufficient_funds",
    "is_first_txn",
]

CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


def engineer_features_batch(data: pd.DataFrame) -> pd.DataFrame:
    """Batch, leakage-safe feature engineering used for training.

    Mirrors the notebook's logic exactly: for every transaction, only
    information from *earlier* transactions by the same sender (`nameOrig`)
    is used. Nothing here looks at the current or future transaction.
    """
    df = data.copy()
    df = df.sort_values(["nameOrig", "step"]).reset_index(drop=True)

    g = df.groupby("nameOrig", sort=False)

    df["hist_per_sender"] = g.cumcount()
    df["prev_step"] = g["step"].shift(1)
    df["is_first_txn"] = df["prev_step"].isna().astype(int)
    df["time_since_last"] = (df["step"] - df["prev_step"]).fillna(0)

    cum_amount_including_current = g["amount"].cumsum()
    df["total_past_amount"] = cum_amount_including_current - df["amount"]

    df["AVG_txn_amount"] = np.where(
        df["hist_per_sender"] > 0,
        df["total_past_amount"] / df["hist_per_sender"],
        0,
    )

    df["balance_change"] = df["oldbalanceOrg"] - df["amount"]
    df["insufficient_funds"] = (df["oldbalanceOrg"] < df["amount"]).astype(int)

    return df


@dataclass
class _SenderRecord:
    count: int = 0
    last_step: int = 0
    total_amount: float = 0.0


class SenderHistoryStore:
    """In-memory, per-sender running statistics for online feature serving.

    Not thread-safe, not persistent, not shared across processes.
    Swap this out for Redis (or similar) before using in a real
    multi-replica production deployment.
    """

    def __init__(self) -> None:
        self._history: Dict[str, _SenderRecord] = {}

    def get_features(self, name_orig: str, step: int, amount: float, old_balance_org: float) -> dict:
        record = self._history.get(name_orig)

        if record is None:
            hist_per_sender = 0
            time_since_last = 0
            is_first_txn = 1
            total_past_amount = 0.0
            avg_txn_amount = 0.0
        else:
            hist_per_sender = record.count
            time_since_last = step - record.last_step
            is_first_txn = 0
            total_past_amount = record.total_amount
            avg_txn_amount = (
                total_past_amount / hist_per_sender if hist_per_sender > 0 else 0.0
            )

        balance_change = old_balance_org - amount
        insufficient_funds = int(old_balance_org < amount)

        return {
            "hist_per_sender": hist_per_sender,
            "time_since_last": time_since_last,
            "is_first_txn": is_first_txn,
            "total_past_amount": total_past_amount,
            "AVG_txn_amount": avg_txn_amount,
            "balance_change": balance_change,
            "insufficient_funds": insufficient_funds,
        }

    def update(self, name_orig: str, step: int, amount: float) -> None:
        """Call AFTER scoring a transaction, so the next transaction from
        the same sender sees this one in its history."""
        record = self._history.setdefault(name_orig, _SenderRecord(last_step=step))
        record.total_amount += amount
        record.count += 1
        record.last_step = step

    def __len__(self) -> int:
        return len(self._history)


def build_feature_row(raw_transaction: dict, engineered: dict) -> pd.DataFrame:
    """Combine a raw transaction with its engineered features into a
    single-row DataFrame matching FEATURE_COLUMNS, ready for the model."""
    row = {**raw_transaction, **engineered}
    return pd.DataFrame([{col: row[col] for col in FEATURE_COLUMNS}])
