from src.train import FEATURES, TARGET, synthetic


def test_no_leakage_column():
    assert "time" not in FEATURES


def test_synthetic_schema():
    df = synthetic()
    assert list(df.columns) == FEATURES + [TARGET]
