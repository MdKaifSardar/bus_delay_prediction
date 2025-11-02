# Flask XGBoost prediction API

This small Flask app exposes a single POST endpoint `/predict` which accepts JSON data and returns predictions from an XGBoost model saved as `best_xgb_model.pkl` in the same directory.

Files added:
- `app.py` — Flask app with `/predict` endpoint.
- `requirements.txt` — Python dependencies.
- `sample_request.json` — example request body.

Usage
1. Install dependencies (prefer a virtualenv):

```powershell
python -m pip install -r requirements.txt
```

2. Place your trained model file named `best_xgb_model.pkl` into the project folder (same directory as `app.py`). The model may be either an sklearn-wrapped XGBoost model (XGBRegressor/XGBClassifier) or a raw `xgboost.Booster` object.

3. Run the app:

```powershell
python app.py
```

4. Example request (single record) — `sample_request.json` contains a placeholder. Use curl or any HTTP client:

```powershell
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @sample_request.json
```

Notes
- The API attempts to align incoming JSON columns to the model's expected `feature_names_in_` if available. If columns are missing they'll be filled with NaN; if extra columns are present they'll be ignored.
- Error responses return JSON with an `error` field and an optional `details` message.
