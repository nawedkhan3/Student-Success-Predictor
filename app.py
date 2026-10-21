import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#0b1020 0%,#101b35 55%,#172554 100%); color:#f8fafc;}
[data-testid="stSidebar"] {background:#0f172a; border-right:1px solid rgba(148,163,184,.22);}
.hero {padding:1.5rem 1.7rem; border:1px solid rgba(148,163,184,.25); border-radius:22px;
background:linear-gradient(120deg,rgba(59,130,246,.24),rgba(14,165,233,.08)); margin-bottom:1.2rem;}
.hero h1 {margin:0; font-size:2.25rem; color:#f8fafc;}
.hero p {margin:.5rem 0 0 0; color:#cbd5e1; font-size:1rem;}
.small-note {color:#94a3b8; font-size:.83rem;}
div.stButton > button[kind="primary"] {border:0; border-radius:12px; font-weight:700; min-height:3rem;
background:linear-gradient(90deg,#2563eb,#06b6d4);}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = Path("student_svm_model.pkl")

st.sidebar.markdown("## 🎓 Student Success")
st.sidebar.caption("Machine Learning • SVM Pipeline")
st.sidebar.divider()
st.sidebar.markdown("### About this project")
st.sidebar.write("Predict a student's exam outcome from study and academic features.")
st.sidebar.info("Educational demo trained on synthetic practice data. Predictions are estimates, not guarantees.")

st.markdown("""
<div class="hero">
<h1>🎓 Student Success Predictor</h1>
<p>Instant exam-outcome predictions powered by your trained SVM Pipeline.</p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.resolve()}")
    st.write("Place `student_svm_model.pkl` in the same folder as this app.py.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error("The model could not be loaded. Check your Python and scikit-learn versions.")
    st.code(str(exc))
    st.stop()

st.markdown("### 🧑‍🎓 Student profile")
st.write("Adjust the details, then predict the outcome.")

with st.form("student_prediction_form"):
    left, right = st.columns(2, gap="large")
    with left:
        age = st.slider("Age", 18, 70, 21)
        study_hours = st.slider("Study hours per day", 0.0, 12.0, 5.0, step=0.5)
        attendance = st.slider("Attendance (%)", 35.0, 100.0, 85.0, step=1.0)
    with right:
        previous_score = st.slider("Previous exam score", 10.0, 100.0, 70.0, step=1.0)
        sleep_hours = st.slider("Sleep hours per day", 3.0, 10.0, 7.0, step=0.5)
        assignments = st.slider("Assignments completed", 0, 10, 8)
    submitted = st.form_submit_button("✨ Predict exam outcome", type="primary", use_container_width=True)

if submitted:
    new_student = pd.DataFrame([{
        "Age": age,
        "Study_Hours_Per_Day": study_hours,
        "Attendance_Percent": attendance,
        "Previous_Exam_Score": previous_score,
        "Sleep_Hours_Per_Day": sleep_hours,
        "Assignments_Completed": assignments
    }])
    try:
        prediction = int(model.predict(new_student)[0])
    except Exception as exc:
        st.error("Prediction failed. Confirm the saved pipeline was trained using these six feature columns.")
        st.code(str(exc))
        st.stop()

    st.divider()
    st.markdown("### 🔮 Model result")
    if prediction == 1:
        st.success("## ✅ Predicted outcome: PASS")
        st.write("The model classifies this profile as **Passed (1)**.")
    else:
        st.warning("## 📘 Predicted outcome: FAIL")
        st.write("The model classifies this profile as **Did not pass (0)**.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Study / day", f"{study_hours:.1f} hrs")
    c2.metric("Attendance", f"{attendance:.0f}%")
    c3.metric("Previous score", f"{previous_score:.0f}/100")
    with st.expander("View model input data"):
        st.dataframe(new_student, use_container_width=True, hide_index=True)

st.divider()
st.markdown('<p class="small-note">Built with Streamlit, scikit-learn Pipeline, StandardScaler and SVM. '
            'Synthetic practice data only; this prediction does not guarantee a real exam outcome.</p>',
            unsafe_allow_html=True)
