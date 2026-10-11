import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
model = joblib.load("student_pass_model_attendance.joblib")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    data = request.get_json(silent=True) or request.values

    try:
        study_hours = float(data.get("study_hours"))
        attendance = float(data.get("attendance_percentage"))
    except (TypeError, ValueError):
        return jsonify(error="study_hours and attendance_percentage must be numbers"), 400

    if study_hours < 0:
        return jsonify(error="study_hours cannot be negative"), 400
    if not 0 <= attendance <= 100:
        return jsonify(error="attendance_percentage must be between 0 and 100"), 400

    features = [[study_hours, attendance]]
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return jsonify(
        study_hours=study_hours,
        attendance_percentage=attendance,
        prediction="Pass" if prediction == 1 else "Fail",
        pass_probability=round(probability, 3),
    )


if __name__ == "__main__":
    app.run(debug=True)
