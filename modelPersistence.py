import joblib
from sklearn.linear_model import LogisticRegression

# Sample training data
# Study hours: 1, 2, 3, 4, 5, 6, 7, 8
# 0 = Fail, 1 = Pass
X = [[1], [2], [3], [4], [5], [6], [7], [8]]
y = [0, 0, 0, 0, 1, 1, 1, 1]

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_pass_model.joblib")

print("Model saved successfully!")

# Load the saved model
loaded_model = joblib.load("student_pass_model.joblib")

# Make three predictions
study_hours = [[2], [5], [8]]
predictions = loaded_model.predict(study_hours)

for hours, prediction in zip(study_hours, predictions):
    result = "Pass" if prediction == 1 else "Fail"
    print(f"Study Hours: {hours[0]}, Prediction: {result}")