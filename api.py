from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("artifacts/model.pkl")
threshold = joblib.load("artifacts/threshold.pkl")


@app.post("/predict")
def predict(data: list):
    x = np.array(data).reshape(1, -1)

    pred = model.predict(x)[0]
    return {
        "value": float(pred),
        "irrigation": int(pred > threshold)
    }