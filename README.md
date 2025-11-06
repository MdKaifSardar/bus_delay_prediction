# Bus Delay Prediction API

This repository implements a small Flask-based prediction service that wraps a pre-trained XGBoost model to predict bus delays for route segments.

The code is organized as a minimal backend package (`backend`) and a tiny entrypoint (`app.py`). The heavy lifting (model loading and prediction) is contained in `backend/model.py` and the HTTP endpoints live in `backend/api.py`.

## Quick overview
- `app.py` — application entrypoint; constructs the Flask app via `backend.create_app()`.
- `backend/__init__.py` — application factory (`create_app`) and background model loader.
- `backend/model.py` — model-loading utilities and a `predict_with_model()` helper.
- `backend/api.py` — Flask blueprint exposing `/predict`, `/ready`, and `/`.

## Model (what the service expects)
- Expected model file: `best_xgb_model.pkl` by default (one level above `backend/`).
- You may override the location by setting the environment variable `MODEL_PATH` to an absolute path.
- The service expects a pickled sklearn-like or XGBoost model. The loader will unpickle the model and, on prediction, will either call `model.predict(df)` (sklearn-like) or convert the input DataFrame to an `xgboost.DMatrix` and call `Booster.predict`.
- The model should accept a pandas DataFrame with the following 15 features (names must match):

  - segment_id
  - distance_km
  - avg_speed_kmph
  - traffic_mean
  - traffic_std
  - traffic_max
  - traffic_p90
  - rain_intensity
  - temperature_celsius
  - visibility_km
  - num_signals
  - num_stops
  - is_holiday
  - day_of_week
  - hour_of_day

Ensure these columns exist and are typed appropriately (numeric where expected). If the model exposes `feature_names_in_`, the API will reindex incoming data to match.

## API

All requests and responses use JSON. The application exposes these endpoints:

- `GET /` — basic info, lists required features and shows an example single-record body.
- `GET /ready` — readiness probe. Returns 200 when the model is loaded; 503 while loading; 500 if loading failed.
- `POST /predict` — run predictions. Accepts either a single JSON object (one record) or a list of objects (batch). Returns:

```json
{ "predictions": [ ... ] }
```

Example single-record request body:

```json
{
  "segment_id": 3,
  "distance_km": 1.60,
  "avg_speed_kmph": 30.03,
  "traffic_mean": 1.48,
  "traffic_std": 0.14,
  "traffic_max": 1.82,
  "traffic_p90": 1.70,
  "rain_intensity": 2.53,
  "temperature_celsius": 28.52,
  "visibility_km": 6.07,
  "num_signals": 0,
  "num_stops": 2,
  "is_holiday": 1,
  "day_of_week": 6,
  "hour_of_day": 13
}
```

Response example:

```json
{ "predictions": [12.3] }
```

HTTP status guidance:
- `200` — success
- `400` — bad request (invalid JSON / unsupported payload)
- `503` — service not ready (model still loading)
- `500` — internal error or model load failure

## Deployment notes (Render)

- The app includes a `Procfile` configured to run an ASGI server with Uvicorn (`uvicorn asgi:app --host 0.0.0.0 --port $PORT`). Ensure `uvicorn` and `asgiref` are in `requirements.txt` (they are).
- The app starts quickly and loads the model in a background thread. Use the `/ready` endpoint as the readiness probe in your Render service settings to avoid routing traffic before the model is ready.
- Model placement options:
  1. Commit `best_xgb_model.pkl` into the repository (not recommended for large or sensitive models). If you choose this, add it and push.
  2. Download the model during Render build (recommended). Configure a build command in Render that downloads the model to the project root, e.g.:

```bash
curl -fSL -o best_xgb_model.pkl "https://<your-storage>/best_xgb_model.pkl"
```

  3. Place the model on the instance and set `MODEL_PATH` to its absolute path.

## Local development

1. Create a virtualenv and install dependencies:

```powershell
pip install -r requirements.txt
```

2. Run the app locally (dev server):

```powershell
python app.py
# or run via uvicorn (ASGI wrapper)
uvicorn asgi:app --host 0.0.0.0 --port 8000
```

3. Test readiness and prediction endpoints with `curl` or HTTP client.

## Best practices & performance tips

- Load the model at build time if possible so runtime startup is fast.
- For high throughput, consider saving XGBoost in native format (`Booster.save_model`) or exporting to ONNX for faster inference.
- Use batching for prediction when possible and consider a dedicated model-serving process if inference is the bottleneck.
- Tune worker count based on memory; if using Gunicorn prefer `--preload` so the master loads the model and forks workers to take advantage of copy-on-write.

## Troubleshooting

- If `/ready` returns 500, check logs for `LOAD_ERROR` details (application logs include unpickle and IO stack traces).
- If predictions return NaN or fail, verify incoming JSON includes all required features and types.

## License & contributions

This project is provided as-is. Contributions are welcome — open a PR with changes or improvements.
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
