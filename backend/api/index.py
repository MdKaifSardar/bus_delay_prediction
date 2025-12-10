from flask import jsonify
from . import bp


@bp.route("/", methods=["GET"])
def index():
    return (
        jsonify(
            {
                "message": "Bus delay API running",
                "endpoints": ["/predict"],
                # Keep the required features and example inline for easy discovery
                "required_features": [
                    "segment_id",
                    "distance_km",
                    "avg_speed_kmph",
                    "traffic_mean",
                    "traffic_std",
                    "traffic_max",
                    "traffic_p90",
                    "rain_intensity",
                    "temperature_celsius",
                    "visibility_km",
                    "num_signals",
                    "num_stops",
                    "is_holiday",
                    "day_of_week",
                    "hour_of_day",
                ],
                "example_single_record": {
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
                    "hour_of_day": 13,
                },
                "notes": [
                    "POST JSON to /predict with Content-Type: application/json.",
                    "Accepts either a single object (as shown) or a list of such objects for batch predictions.",
                    "If the model exposes feature_names_in_, the server will reindex incoming data to that order; missing features become NaN (may cause prediction to fail).",
                ],
            }
        ),
        200,
    )
