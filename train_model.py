import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# ---- Training data (replace with your own dataset) ----
# study_hours, attendance_percentage -> passed (1) / failed (0)
X_full = np.array([
    [1, 50], [2, 55], [3, 60], [4, 70], [5, 75],
    [6, 80], [7, 90], [8, 92], [2, 95], [9, 40],
    [1, 85], [10, 98], [3, 45], [6, 65], [4, 50],
])
y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0])

# ---- Exercise 1: model using study hours only ----
hours_model = LogisticRegression()
hours_model.fit(X_full[:, [0]], y)
joblib.dump(hours_model, "student_pass_model.joblib")
print("Saved student_pass_model.joblib (study hours only)")

# ---- Exercise 2: retrained model using hours + attendance ----
full_model = LogisticRegression()
full_model.fit(X_full, y)
joblib.dump(full_model, "student_pass_model_attendance.joblib")
print("Saved student_pass_model_attendance.joblib (hours + attendance)")
