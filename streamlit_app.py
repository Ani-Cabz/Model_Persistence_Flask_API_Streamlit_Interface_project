
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Application title
st.title("🎓 Student Performance Predictor")
st.write(
    "Enter the student's name and daily study hours "
    "to predict the student's academic performance."
)

# Load the saved machine learning model
MODEL_PATH = Path(__file__).resolve().parent / "student_pass_model.joblib"

if not MODEL_PATH.exists():
    st.error(
        "Model file not found: student_pass_model.joblib"
    )
    st.info(
        "Please run modelPersistence.py first to create "
        "the saved model in the same folder."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

# Student information inputs
st.subheader("Student Information")

student_name = st.text_input(
    "Student Name",
    placeholder="Enter student name"
)

study_hours = st.number_input(
    "Study Hours per Day",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

# Generate prediction
if st.button("Predict Student Performance", type="primary"):
    if not student_name.strip():
        st.warning("Please enter the student's name.")
    else:
        # Predict using the trained model
        prediction = int(model.predict([[study_hours]])[0])

        # Display prediction result
        st.subheader("Prediction Result")
        st.write(f"**Student Name:** {student_name.strip()}")
        st.write(f"**Study Hours:** {study_hours} hours per day")

        if prediction == 1:
            st.success("🎉 Prediction: PASS")
        else:
            st.error("Prediction: FAIL")

        # Explain how the prediction was generated
        st.subheader("How Was the Prediction Generated?")

        st.write(
            "The system uses a trained Logistic Regression "
            "machine learning model. The model receives the "
            "student's study hours as input and uses patterns "
            "learned from the training data to predict the result."
        )

        st.write(
            "**Class 0 = Fail** | **Class 1 = Pass**"
        )

        st.caption(
            "Note: This is an educational prediction based on "
            "sample training data. It is not a guarantee of "
            "actual academic performance."
        )

        # Bonus: Chart comparing study hours and predicted class
        st.subheader("Study Hours vs. Predicted Class")

        chart_data = pd.DataFrame({
            "Metric": ["Study Hours", "Predicted Class"],
            "Value": [study_hours, prediction]
        })

        fig, ax = plt.subplots(figsize=(7, 4))

        bars = ax.bar(
            chart_data["Metric"],
            chart_data["Value"]
        )

        ax.set_title(f"Prediction for {student_name.strip()}")
        ax.set_ylabel("Value")
        ax.set_ylim(0, max(2, study_hours + 1))

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:g}",
                ha="center",
                va="bottom"
            )

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            f"Predicted class: {prediction} "
            f"({'Pass' if prediction == 1 else 'Fail'})"
        )
