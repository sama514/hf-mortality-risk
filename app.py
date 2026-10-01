import joblib
import pandas as pd
import streamlit as st

st.title("Heart failure: mortality risk estimate")
st.warning("Educational research demo. NOT a diagnostic or clinical decision tool.")
art = joblib.load("reports/model.joblib")
v = {
    "age": st.slider("Age", 40, 95, 60),
    "ejection_fraction": st.slider("Ejection fraction (%)", 14, 80, 38),
    "serum_creatinine": st.slider("Serum creatinine (mg/dL)", 0.5, 9.5, 1.1),
    "serum_sodium": st.slider("Serum sodium (mEq/L)", 113, 148, 137),
    "creatinine_phosphokinase": st.number_input("CPK (mcg/L)", 20, 8000, 250),
    "platelets": st.number_input("Platelets (kiloplatelets/mL)", 25000, 850000, 260000),
    "anaemia": int(st.checkbox("Anaemia")), "diabetes": int(st.checkbox("Diabetes")),
    "high_blood_pressure": int(st.checkbox("High blood pressure")),
    "sex": int(st.selectbox("Sex", ["Female", "Male"]) == "Male"),
    "smoking": int(st.checkbox("Smoker")),
}
p = art["model"].predict_proba(pd.DataFrame([v])[art["features"]])[0, 1]
st.metric("Estimated risk", f"{p:.0%}")
st.write("Above operating threshold" if p >= art["threshold"] else "Below operating threshold")
st.caption("Trained on ~300 patients from a single hospital. See README > Limitations.")
