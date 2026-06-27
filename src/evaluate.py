import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "rmse": rmse,
        "r2": r2
    }


def classification_metrics(y_true_bin, y_score, threshold):
    y_pred = (y_score > threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true_bin == 1))
    fp = np.sum((y_pred == 1) & (y_true_bin == 0))
    fn = np.sum((y_pred == 0) & (y_true_bin == 1))
    tn = np.sum((y_pred == 0) & (y_true_bin == 0))

    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "accuracy": accuracy
    }