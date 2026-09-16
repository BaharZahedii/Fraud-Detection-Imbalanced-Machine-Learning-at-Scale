import pandas as pd

from src.features import SenderHistoryStore, engineer_features_batch


def test_sender_history_store_first_transaction_is_cold_start():
    store = SenderHistoryStore()
    feats = store.get_features(name_orig="A1", step=10, amount=100.0, old_balance_org=100.0)

    assert feats["hist_per_sender"] == 0
    assert feats["is_first_txn"] == 1
    assert feats["total_past_amount"] == 0.0
    assert feats["AVG_txn_amount"] == 0.0


def test_sender_history_store_tracks_second_transaction():
    store = SenderHistoryStore()
    store.update(name_orig="A1", step=10, amount=100.0)

    feats = store.get_features(name_orig="A1", step=15, amount=50.0, old_balance_org=200.0)

    assert feats["hist_per_sender"] == 1
    assert feats["is_first_txn"] == 0
    assert feats["time_since_last"] == 5
    assert feats["total_past_amount"] == 100.0
    assert feats["AVG_txn_amount"] == 100.0


def test_insufficient_funds_flag():
    store = SenderHistoryStore()
    feats = store.get_features(name_orig="A2", step=1, amount=500.0, old_balance_org=100.0)
    assert feats["insufficient_funds"] == 1

    feats_ok = store.get_features(name_orig="A3", step=1, amount=50.0, old_balance_org=100.0)
    assert feats_ok["insufficient_funds"] == 0


def test_engineer_features_batch_matches_online_store_for_same_sender():
    """The batch (training-time) and online (serving-time) feature
    computations must agree, or the model will see a distribution shift
    between training and serving."""
    df = pd.DataFrame(
        [
            {"nameOrig": "A1", "step": 1, "amount": 100.0, "oldbalanceOrg": 100.0, "isFraud": 0},
            {"nameOrig": "A1", "step": 5, "amount": 40.0, "oldbalanceOrg": 60.0, "isFraud": 0},
            {"nameOrig": "A1", "step": 9, "amount": 20.0, "oldbalanceOrg": 30.0, "isFraud": 0},
        ]
    )
    batch = engineer_features_batch(df)

    store = SenderHistoryStore()
    online_rows = []
    for _, row in batch.sort_values("step").iterrows():
        feats = store.get_features(
            name_orig=row["nameOrig"], step=row["step"], amount=row["amount"], old_balance_org=row["oldbalanceOrg"]
        )
        online_rows.append(feats)
        store.update(row["nameOrig"], row["step"], row["amount"])

    for i, feats in enumerate(online_rows):
        batch_row = batch.iloc[i]
        assert feats["hist_per_sender"] == batch_row["hist_per_sender"]
        assert feats["is_first_txn"] == batch_row["is_first_txn"]
        assert feats["total_past_amount"] == batch_row["total_past_amount"]
        assert feats["AVG_txn_amount"] == batch_row["AVG_txn_amount"]
