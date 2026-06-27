## Smart Irrigation Prediction using Random Forest
## About the Project
This project predicts irrigation requirements in tomato cultivation using IoT sensor data and a Random Forest machine learning model.

The model estimates the amount of water required and classifies whether irrigation is needed based on environmental and soil measurements.

## Main Features
1. Feature engineering
2. Time-based train/validation/test split
3. Random Forest Regressor
4. Hyperparameter tuning
5. Threshold optimization
6. Performance evaluation using RMSE, R2, Precision, Recall and F1-score

## Dataset

This project uses the dataset:

**IoT-based Data Collection in a Tomato Cultivation Under Different Irrigation Regimes**

🔗 https://data.mendeley.com/datasets/35wh56287y/2

The dataset contains sensor measurements collected from a smart greenhouse, including:
```bash
1. Air temperature
2. Air humidity
3. Soil temperature
4. Soil moisture
5. Solar radiation
6. Wind speed
7. Electrical conductivity
8. CO2 concentration
9. Water consumption
```
## External Data Sources

In addition to the main dataset collected from the experimental greenhouse in Parma, Italy, additional meteorological data was integrated using the NASA POWER API.

This data was used to enrich the dataset with environmental variables such as:

1. Solar radiation
2. Wind speed
3. Temperature
4. Humidity estimates

The integration helped improve model performance by providing broader climate context for irrigation prediction.

## 1. Installation
```bash
git clone https://github.com/BurakYildirim-cmd/random-forest-irrigation.git
cd tomato-irrigation-RF
```
## 2. Create a virtual environment (recommended)
Windows
```bash
python -m venv venv venv\Scripts\activate
```
Linux / macOS
```bash
python3 -m venv venv
source venv/bin/activate
```
## 3. Install dependencies
```bash
pip install -r requirements.txt
```
## 4. Train the model
```bash
python src/train.py
```
After training, the following files will be generated automatically:

1. artifacts/model.pkl
2. artifacts/threshold.pkl

## 6. Run the API (optional)
```bash
uvicorn api:app --reload
```
Open your browser at:
```bash
http://127.0.0.1:8000/docs
```

to test the prediction API using the interactive Swagger interface.
