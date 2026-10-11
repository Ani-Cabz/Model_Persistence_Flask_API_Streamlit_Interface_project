import joblib

# Load the saved model: no training happens here.
model = joblib.load("student_pass_model.joblib")

for hours in [2, 5, 8]:
    pred = model.predict([[hours]])[0]
    prob = model.predict_proba([[hours]])[0][1]
    label = "Pass" if pred == 1 else "Fail"
    print(f"{hours} study hours -> {label} (pass probability {prob:.2f})")
