import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. Basic configuration ====================
model = joblib.load('xgboost_final_model.pkl')
X_test = pd.read_csv('dataxgboost.csv')

feature_names = ["d3_a_ado2", "d3_lac", "d3_sofa", "peep"]
feature_labels = {
    "d3_a_ado2": "D3 A-aDO2 (mmHg)",
    "d3_lac":    "D3 LAC (mmol/L)",
    "d3_sofa":   "D3 SOFA score",
    "peep":      "PEEP (cmH2O)"
}

# ==================== 2. Streamlit page configuration ====================
st.set_page_config(page_title="ARDS Mortality Risk Predictor", layout="wide")
st.title("ARDS 28-Day Mortality Risk Predictor")
st.markdown("### Enter the patient's Day-3 metrics below and click Predict")

# ==================== 3. Feature input components ====================
d3_a_ado2 = st.number_input(feature_labels["d3_a_ado2"], min_value=150.0, max_value=660.0, value=280.0, step=1.0)
d3_lac    = st.number_input(feature_labels["d3_lac"],    min_value=0.3,  max_value=12.0,  value=1.4,   step=0.1)
d3_sofa   = st.number_input(feature_labels["d3_sofa"],   min_value=1,    max_value=20,    value=8,     step=1)
peep      = st.number_input(feature_labels["peep"],      min_value=0,    max_value=28,    value=6,     step=1)

# ==================== 4. Data processing and prediction ====================
feature_values = [d3_a_ado2, d3_lac, d3_sofa, peep]
features_df = pd.DataFrame([feature_values], columns=feature_names)

if st.button("Predict", type="primary"):
    predicted_class = int(model.predict(features_df)[0])
    predicted_proba = model.predict_proba(features_df)[0]

    # --- 结果展示 ---
    st.subheader("Prediction Results")
    risk_label = "High risk" if predicted_class == 1 else "Low risk"
    st.write(f"**Risk class: {predicted_class} ({risk_label})**")
    st.write(f"**Probability:** Survival {predicted_proba[0]:.2%} | Mortality {predicted_proba[1]:.2%}")

    st.subheader("Clinical Note")
    mortality_probability = predicted_proba[1] * 100
    if predicted_class == 1:
        advice = (
            f"The model predicts a high 28-day mortality risk ({mortality_probability:.1f}%). "
            "Close monitoring of hemodynamics, oxygenation, and organ function is recommended; "
            "consider escalation of respiratory support or ICU transfer as clinically indicated."
        )
    else:
        advice = (
            f"The model predicts a relatively low 28-day mortality risk ({mortality_probability:.1f}%). "
            "Continue current management and reassess relevant metrics dynamically."
        )
    st.info(advice)

    # ==================== 5. SHAP explanation ====================
    st.subheader("SHAP Feature Contribution")

    explainer = shap.TreeExplainer(
        model,
        data=X_test[feature_names],
        model_output='probability'
    )
    shap_values = explainer(features_df)   # Explanation, values.shape = (1, 4, 2)

    # ---------- 关键修复：取单个样本 + 正类（索引 1）----------
    vals = shap_values.values[0]        # (4, 2) 或 (4,)
    base = shap_values.base_values[0]   # (2,)  或标量

    if vals.ndim == 2:                  # 多输出二分类
        vals = vals[:, 1]               # 取正类（Mortality），→ (4,)
        if np.ndim(base) > 0:
            base = base[1]              # 取正类的 base value（标量）
    # -------------------------------------------------------

    exp = shap.Explanation(
        values=vals,
        base_values=base,
        data=features_df.iloc[0].values,
        feature_names=[feature_labels[f] for f in feature_names]
    )

    # 可选：保留这段 DEBUG，确认形状正确后再删
    # st.write("DEBUG values.shape =", exp.values.shape)   # 应为 (4,)
    # st.write("DEBUG base_values    =", exp.base_values)   # 应为标量

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=10, show=False)
    st.pyplot(plt.gcf())
    plt.close()
