import numpy as np
import joblib
from itertools import product

from src.data import load_data, split_timewise
from src.features import add_features
from src.model import get_model
from src.evaluate import regression_metrics, classification_metrics


DATA_PATH = "data/dataset.csv"


def find_best_threshold(y_true_bin, y_score):
    thresholds = np.linspace(y_score.min(), y_score.max(), 50)

    best_score = -1
    best_thr = 0

    for t in thresholds:
        metrics = classification_metrics(y_true_bin, y_score, t)

        score = (
            metrics["f1"]
            + 0.1 * metrics["recall"]
            + 0.1 * metrics["precision"]
        )

        if score > best_score:
            best_score = score
            best_thr = t

    return best_thr


def main():

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Feature engineering...")
    df = add_features(df)

    df = df.dropna()

    # FEATURES (target hariç)
    features = [c for c in df.columns if c not in ["instant_water", "timestamp"]]

    train_df, val_df, test_df = split_timewise(df)

    X_train = train_df[features]
    y_train = train_df["instant_water"]
    y_train_bin = (y_train > 0).astype(int)

    X_val = val_df[features]
    y_val = val_df["instant_water"]
    y_val_bin = (y_val > 0).astype(int)

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt", 0.7],
    }

    best_model = None
    best_val_score = -1
    best_thr = 0

    print(f"Total configs: {len(list(product(*param_grid.values())))}")

    for combo in product(*param_grid.values()):

        cfg = dict(zip(param_grid.keys(), combo))

        model = get_model(cfg)
        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)

        thr = find_best_threshold(y_val_bin.values, val_pred)

        cls_metrics = classification_metrics(y_val_bin.values, val_pred, thr)
        reg_metrics = regression_metrics(y_val, val_pred)

        # Kombine skor
        score = cls_metrics["f1"] + 0.1 * cls_metrics["recall"] + 0.1 * cls_metrics["precision"]

        if score > best_val_score:
            best_val_score = score
            best_model = model
            best_thr = thr

            print("\nNEW BEST MODEL")
            print(cfg)
            print(cls_metrics)
            print(reg_metrics)

    # FINAL TRAIN (train + val)
    train_val = train_df.append(val_df)

    X_train_val = train_val[features]
    y_train_val = train_val["instant_water"]

    final_model = get_model(cfg)
    final_model.fit(X_train_val, y_train_val)

    # SAVE ARTIFACTS
    joblib.dump(final_model, "artifacts/model.pkl")
    joblib.dump(best_thr, "artifacts/threshold.pkl")

    print("\nTraining completed!")
    print("Best threshold:", best_thr)


if __name__ == "__main__":
    main()