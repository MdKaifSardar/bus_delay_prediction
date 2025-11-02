from flask import Flask, request, jsonify
import os
import pickle
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

MODEL_FILENAME = os.path.join(os.path.dirname(__file__), "best_xgb_model.pkl")


def load_model(path=MODEL_FILENAME):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


# Try to load model at import time; keep error message if loading failed so endpoint can return 500
try:
    model = load_model()
    load_error = None
except Exception as e:
    model = None
    load_error = str(e)


@app.route("/predict", methods=["POST"])
def predict():
    global model, load_error
    # If model didn't load at startup, return an error
    if model is None:
        return jsonify({"error": "Model not loaded", "details": load_error}), 500

    # Parse JSON body
    try:
        payload = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "Invalid JSON", "details": str(e)}), 400

    if payload is None:
        return jsonify({"error": "Empty request body"}), 400

    # Convert incoming JSON into a pandas DataFrame. Support a single record (dict) or
    # a list of records.
    try:
        if isinstance(payload, dict):
            # If values are scalars -> single row; if values are lists -> DataFrame from dict
            if all(not isinstance(v, (list, tuple)) for v in payload.values()):
                df = pd.DataFrame([payload])
            else:
                df = pd.DataFrame(payload)
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            return jsonify({"error": "Unsupported JSON format"}), 400
    except Exception as e:
        return jsonify({"error": "Failed to convert JSON to DataFrame", "details": str(e)}), 400

    # Align DataFrame columns to what the model expects when possible
    try:
        # sklearn-style wrappers (XGBClassifier/Regressor) often expose feature_names_in_
        if hasattr(model, "feature_names_in_"):
            required = list(model.feature_names_in_)
            df = df.reindex(columns=required)
        else:
            # For raw xgboost.Booster we'll pass the df as-is and construct a DMatrix
            pass
    except Exception as e:
        return jsonify({"error": "Error during preprocessing", "details": str(e)}), 500

    # Try to coerce numeric columns where appropriate
    try:
        df = df.apply(pd.to_numeric, errors='ignore')
    except Exception:
        # Non-fatal: continue with original df if conversion fails
        pass

    # Run prediction for sklearn-like objects or raw Booster
    try:
        # sklearn wrapper
        if hasattr(model, "predict") and "Booster" not in type(model).__name__:
            preds = model.predict(df)
        else:
            # assume xgboost.core.Booster
            dmat = xgb.DMatrix(df)
            preds = model.predict(dmat)

        # Convert predictions to plain python list for JSON serialization
        if hasattr(preds, "tolist"):
            out = preds.tolist()
        else:
            out = [float(preds)]

        return jsonify({"predictions": out})
    except Exception as e:
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


if __name__ == "__main__":
    # Simple local run for development
    app.run(host="0.0.0.0", port=5000, debug=True)
