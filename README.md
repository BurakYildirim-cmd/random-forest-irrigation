## Installation
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
