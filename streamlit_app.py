import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Pass Predictor", page_icon="🎓")
st.title("🎓 Student Pass Predictor")

@st.cache_resource
def load_model():
    return joblib.load("student_pass_model.joblib")

model = load_model()

name = st.text_input("Student name")
hours = st.number_input("Study hours per day", min_value=0.0, max_value=16.0, value=3.0, step=0.5)

if st.button("Predict"):
    if not name.strip():
        st.warning("Please enter the student's name.")
    else:
        pred = int(model.predict([[hours]])[0])
        prob = float(model.predict_proba([[hours]])[0][1])

        if pred == 1:
            st.success(f"{name} is predicted to PASS (probability {prob:.0%}).")
        else:
            st.error(f"{name} is predicted to FAIL (pass probability {prob:.0%}).")

        st.info(
            "How this was generated: the model was trained earlier on past student "
            "data and loaded from a saved file. It takes your study hours, computes a "
            "pass probability with logistic regression, and predicts Pass when that "
            "probability is 50% or higher."
        )

        # Bonus: study hours vs predicted class
        hrs_range = [h / 2 for h in range(0, 33)]
        curve = pd.DataFrame(
            {"Pass probability": [model.predict_proba([[h]])[0][1] for h in hrs_range]},
            index=pd.Index(hrs_range, name="Study hours"),
        )
        st.subheader("Study hours vs. predicted pass probability")
        st.line_chart(curve)
        st.caption(f"{name}: {hours} hours, predicted {'Pass' if pred else 'Fail'}")
