
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    # Validate JSON input
    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be valid JSON."
        }), 400

    # Accept both inputs
    study_hours = data.get("study_hours")
    attendance_percentage = data.get("attendance_percentage")

    # Validate required inputs
    if study_hours is None or attendance_percentage is None:
        return jsonify({
            "error": "Both study_hours and attendance_percentage are required."
        }), 400

    # Validate numeric values
    try:
        study_hours = float(study_hours)
        attendance_percentage = float(attendance_percentage)
    except (ValueError, TypeError):
        return jsonify({
            "error": "Both inputs must be numbers."
        }), 400

    # Validate attendance range
    if not 0 <= attendance_percentage <= 100:
        return jsonify({
            "error": "Attendance percentage must be between 0 and 100."
        }), 400

    # Predict using study hours only
    prediction = model.predict([[study_hours]])[0]

    result = "Pass" if prediction == 1 else "Fail"

    return jsonify({
        "study_hours": study_hours,
        "attendance_percentage": attendance_percentage,
        "prediction": result
    })


if __name__ == "__main__":
    app.run(debug=True)
