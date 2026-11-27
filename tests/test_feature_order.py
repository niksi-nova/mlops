"""Fails if the model input column order is wrong."""
import joblib
import pytest

from src.models.evaluate import MODELS_DIR, load_feature_spec, load_model_index, load_split, predict
from src.models.features import model_feature_columns, score
from src.monitoring.performance import BASELINE_TEST_MAE, mae


@pytest.fixture(scope="module")
def setup():
    try:
        spec = load_feature_spec()
        test = load_split("test")
        info = load_model_index()["lightgbm_l1"]
        model = joblib.load(MODELS_DIR / info["file"])
    except Exception as exc:  # data or model not pulled
        pytest.skip("needs DVC data and models: %s" % exc)
    return spec, test, model


def test_order_is_categoricals_then_numerics(setup):
    spec, _, model = setup
    cols = model_feature_columns(model, spec)
    assert cols == spec["categorical_features"] + spec["numeric_features"]
    assert cols != spec["features"]  # the spec order is the wrong one


def test_clean_test_mae_is_about_30_6(setup):
    spec, test, model = setup
    pred = score("lightgbm_l1", model, test, spec)
    assert abs(mae(test["travel_time_s"], pred) - BASELINE_TEST_MAE) < 0.2


def test_spec_order_is_clearly_worse(setup):
    spec, test, model = setup
    bad = predict("lightgbm_l1", model, test, spec["features"], spec["categorical_features"])
    assert mae(test["travel_time_s"], bad) > 100
