"""Build reports/drift_reference.json from TRAINING data (days 1-61). No W&B, no registry."""
import joblib
from src.models import evaluate as ev
from src.monitoring.reference import build_reference, save_reference

DRIFT_N_BINS = 10


def main() -> None:
    spec = ev.load_feature_spec()
    train = ev.load_split("train")
    train = train[(train["day_index"] >= 1) & (train["day_index"] <= 61)]

    index = ev.load_model_index()
    model = joblib.load(ev.MODELS_DIR / index["lightgbm_l1"]["file"])
    cols = model.booster_.feature_name()  # model's own training column order
    preds = ev.predict("lightgbm_l1", model, train, cols, spec["categorical_features"])

    ref = build_reference(
        train, spec["numeric_features"], spec["categorical_features"],
        spec["target"], preds, DRIFT_N_BINS,
    )
    ref["dataset_version"] = "delhi-data-v2"
    ref["split"] = "train days 1-61"
    save_reference(ref)
    print("rows:", ref["n_rows"], "| features:", len(ref["features"]))
    print("target mean:", round(ref["target"]["mean"], 2), "| prediction mean:", round(ref["prediction"]["mean"], 2))


if __name__ == "__main__":
    main()
