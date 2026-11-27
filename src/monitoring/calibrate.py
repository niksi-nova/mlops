"""Estimate performance limits from complete validation windows only."""
import joblib
import numpy as np

from src.models.evaluate import MODELS_DIR, load_feature_spec, load_model_index, load_split
from src.models.features import score
from src.monitoring.run_drift_demo import MODEL_NAME, load_params


def main():
    days = load_params()["window_days"]
    frame = load_split("val")
    model = joblib.load(MODELS_DIR / load_model_index()[MODEL_NAME]["file"])
    pred = score(MODEL_NAME, model, frame, load_feature_spec())
    errors = (frame["travel_time_s"] - pred).abs()
    baseline = errors.mean()
    windows = (frame.day_index - frame.day_index.min()) // days
    changes = []
    for _, group in frame.groupby(windows):
        if group.day_index.nunique() == days:
            value = errors.loc[group.index].mean()
            changes.append(100 * (value / baseline - 1))
            print(f"Days {group.day_index.min()}–{group.day_index.max()}  MAE {value:.3f} s")
    mean, sd = np.mean(changes), np.std(changes, ddof=1)
    print(f"Validation MAE {baseline:.4f} s, mean change {mean:.4f}%, SD {sd:.4f} points")
    for label, multiplier in [("WARNING", 2), ("ALERT", 3)]:
        print(f"{label} {mean + multiplier * sd:.2f}%")


if __name__ == "__main__":
    main()
