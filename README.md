# Bus Delay Prediction Backend

Production-ready Flask/ASGI service that wraps a pre-trained XGBoost model to predict bus delays for individual route segments. The project ships with a clean package layout, background model loading, readiness probes, Docker packaging, and deployment guidance.

## Features
- **15-feature inference schema** with automatic column alignment and numeric coercion.
- **Non-blocking startup**: the model loads in a background thread; `/ready` reports status.
- **Universal prediction helper** works with sklearn wrappers or raw `xgboost.Booster`.
- **ASGI + WSGI support**: develop with Flask, deploy with Uvicorn/Gunicorn.
- **Container-first workflow**: Dockerfile + push-to-Docker-Hub instructions, ready for Render/Heroku/etc.

## Repository Layout
```
app.py                 # Minimal Flask entrypoint
asgi.py                # WSGI → ASGI adapter for Uvicorn
backend/
  __init__.py          # create_app + background loader
  api/
    __init__.py        # Blueprint wiring
    index.py           # GET /
    predict.py         # POST /predict
    health.py          # GET /ready, favicon
  models/
    __init__.py        # Re-export utils loaders/predictors
  services/
    predict.py         # Service layer orchestration
  utils/
    loader.py          # Model path resolution + caching
    predict.py         # DataFrame prep + prediction helpers
    delay_prediction.py# Legacy import shim
Dockerfile
requirements.txt
```

## Requirements
- Python 3.11+
- pipenv/venv recommended
- Model artifact: `backend/models/best_xgb_model.pkl` (or set `MODEL_PATH`)
- System libs: handled by Dockerfile (`build-essential`).

## Environment Variables
| Name | Default | Purpose |
|------|---------|---------|
| `MODEL_PATH` | `backend/models/best_xgb_model.pkl` relative to repo | Override model location (absolute path recommended in prod). |
| `PORT` | `8000` (Dockerfile) | Exposed port for Uvicorn. |
| `FLASK_ENV` | `development` (optional) | Enables debug auto-reload when running `python app.py`. |

## Local Development
1. **Create & activate a virtualenv**
   ```powershell
   cd "E:\CSE\bus delay backend"
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Place the model** at `backend\models\best_xgb_model.pkl` or set `$env:MODEL_PATH`.
4. **Run the dev server (WSGI)**
   ```powershell
   python app.py
   ```
   or run the ASGI stack (recommended):
   ```powershell
   uvicorn asgi:app --host 0.0.0.0 --port 5000
   ```
5. **Smoke-test**
   ```powershell
   Invoke-RestMethod `
     -Uri "http://localhost:5000/predict" `
     -Method Post `
     -ContentType "application/json" `
     -Body '{
       "segment_id":3,
       "distance_km":1.6,
       "avg_speed_kmph":30.03,
       "traffic_mean":1.48,
       "traffic_std":0.14,
       "traffic_max":1.82,
       "traffic_p90":1.7,
       "rain_intensity":2.53,
       "temperature_celsius":28.52,
       "visibility_km":6.07,
       "num_signals":0,
       "num_stops":2,
       "is_holiday":1,
       "day_of_week":6,
       "hour_of_day":13
     }'
   ```

## API Reference
| Endpoint | Method | Description | Success Response |
|----------|--------|-------------|------------------|
| `/` | GET | Service metadata, feature list, example payload. | 200 JSON info. |
| `/ready` | GET | Readiness probe. Returns 200 when the background loader finished. | `{ "ready": true }` or error details. |
| `/predict` | POST | Accepts a JSON object or list of objects with the 15 features. | `{ "predictions": [float, ...] }` |

**Prediction schema (required keys)**
1. `segment_id`
2. `distance_km`
3. `avg_speed_kmph`
4. `traffic_mean`
5. `traffic_std`
6. `traffic_max`
7. `traffic_p90`
8. `rain_intensity`
9. `temperature_celsius`
10. `visibility_km`
11. `num_signals`
12. `num_stops`
13. `is_holiday`
14. `day_of_week`
15. `hour_of_day`

The service converts scalars/lists into a pandas `DataFrame`, reorders columns if the model exposes `feature_names_in_`, and coerces numerics per column.

## Docker Workflow
Build locally:
```powershell
docker build -t bus-delay-backend:latest .
```
Run with baked-in model:
```powershell
docker run -p 8000:8000 --rm bus-delay-backend:latest
```
Run while mounting a model file:
```powershell
docker run -p 8000:8000 --rm \
  -v "${PWD}\backend\models\best_xgb_model.pkl:/app/backend/models/best_xgb_model.pkl" \
  -e MODEL_PATH=/app/backend/models/best_xgb_model.pkl \
  bus-delay-backend:latest
```
Published image example:
```powershell
docker pull mdkaif001/bus-delay-backend:1.0.0
docker run -p 8000:8000 --rm mdkaif001/bus-delay-backend:1.0.0
```

## Deployment Tips
- **Render / Heroku**: use `uvicorn asgi:app --host 0.0.0.0 --port $PORT`. Point the health check to `/ready`.
- **Model sourcing**:
  1. Commit the `.pkl` (if allowed).
  2. Download during build (`curl ... > backend/models/best_xgb_model.pkl`).
  3. Mount at runtime + `MODEL_PATH`.
- **XGBoost warning**: If you see the warning about serialized models, re-save the model using the same XGBoost version as production (`Booster.save_model`) or pin `xgboost==<training-version>` in `requirements.txt`.

## Verification & Testing
- Lint / syntax check:
  ```powershell
  python -m compileall -q .
  ```
- Readiness:
  ```powershell
  curl http://localhost:5000/ready
  ```
- API contract tests can be added under `tests/` (not shipped yet). For quick manual validation, use the sample payload above.

## Troubleshooting
- **/ready returns 500**: check container logs; `app.config['LOAD_ERROR']` stores the message (missing/corrupt model, permissions, etc.).
- **/predict returns 503**: call again once `ready` becomes true.
- **FutureWarning from XGBoost**: re-save the model or pin the package version as noted.
- **Docker cannot connect to daemon**: ensure Docker Desktop (or another engine) is running before building.

## Contributing
Pull requests are welcome. Please include:
1. Clear description of changes.
2. Tests or manual verification notes (`curl /ready`, `/predict`).
3. Updated documentation if behavior changes.

---
Need help? Open an issue with logs, reproduction steps, and the payload you used.
