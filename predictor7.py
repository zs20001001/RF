import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（用于 SHAP 图的中文标签）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 基础配置 ====================
# 加载训练好的 CatBoost 模型（确保 model.pkl 与脚本同目录）
model = joblib.load('model.pkl')

# 加载背景数据（用于 SHAP 解释器，确保 datacatboost.csv 与脚本同目录）
X_test = pd.read_csv('datacatboost.csv')

# 定义特征名称（替换为业务相关列名，与模型输入顺序一致）
feature_names = [
    "d3_a_ado2", "d3_lac", "d3_sofa", "peep"
]

# 中文标签映射
feature_labels = {
    "d3_a_ado2": "D3 A-aDO2（肺泡-动脉氧分压差，mmHg）",
    "d3_lac": "D3 LAC（乳酸，mmol/L）",
    "d3_sofa": "D3 SOFA 评分",
    "peep": "PEEP（呼气末正压，cmH2O）"
}

# ==================== 2. Streamlit 页面配置 ====================
st.set_page_config(page_title="ARDS 28天死亡风险预测器", layout="wide")
st.title("🫁 ARDS 患者 28 天死亡风险预测器")
st.markdown("### 请填写患者第 3 天指标，点击预测获取 28 天死亡风险评估结果")

# ==================== 3. 特征输入组件 ====================
d3_a_ado2 = st.number_input(
    feature_labels["d3_a_ado2"],
    min_value=150.0,
    max_value=660.0,
    value=280.0,
    step=1.0
)

d3_lac = st.number_input(
    feature_labels["d3_lac"],
    min_value=0.3,
    max_value=12.0,
    value=1.4,
    step=0.1
)

d3_sofa = st.number_input(
    feature_labels["d3_sofa"],
    min_value=1,
    max_value=20,
    value=8,
    step=1
)

peep = st.number_input(
    feature_labels["peep"],
    min_value=0,
    max_value=28,
    value=6,
    step=1
)

# ==================== 4. 数据处理与预测 ====================
# 整合用户输入特征（顺序必须与 feature_names 一致）
feature_values = [d3_a_ado2, d3_lac, d3_sofa, peep]

# 转换为 DataFrame 格式（自带列名）
features_df = pd.DataFrame([feature_values], columns=feature_names)

# 预测按钮逻辑
if st.button("预测", type="primary"):
    # 模型预测
    predicted_class = int(model.predict(features_df)[0])
    predicted_proba = model.predict_proba(features_df)[0]

    # 显示预测结果（中文适配）
    st.subheader("📊 预测结果")
    risk_label = "高风险" if predicted_class == 1 else "低风险"
    st.write(f"**风险等级：{predicted_class} ({risk_label})**")
    st.write(f"**风险概率：** 存活概率 {predicted_proba[0]:.2%} | 死亡概率 {predicted_proba[1]:.2%}")

    # 生成个性化建议（中文）
    st.subheader("💡 临床提示")
    death_probability = predicted_proba[1] * 100
    if predicted_class == 1:
        advice = (
            f"模型预测该患者 28 天死亡风险为高风险（概率 {death_probability:.1f}%）。\n"
            "建议密切监测血流动力学、氧合及器官功能，及时评估是否需要升级呼吸支持或转入 ICU。\n"
            "同时请结合临床实际，动态复查 A-aDO2、乳酸、SOFA 评分等指标。"
        )
    else:
        advice = (
            f"模型预测该患者 28 天死亡风险相对较低（死亡概率 {death_probability:.1f}%）。\n"
            "请继续当前治疗并密切观察病情变化，定期复评相关指标。"
        )
    st.info(advice)

    # ==================== 5. SHAP 解释 ====================
    st.subheader("🔍 SHAP 特征贡献解释")

    # 构建 TreeExplainer（概率尺度，更易于临床理解）
    explainer = shap.TreeExplainer(model, data=X_test[feature_names], model_output='probability')

    # 计算当前样本的 SHAP 值
    shap_values = explainer(features_df)

    # 构建 Explanation 对象（用于瀑布图）
    exp = shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        data=features_df.iloc[0].values,
        feature_names=[feature_labels[f] for f in feature_names]
    )

    # 绘制瀑布图
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=10, show=False)

    # 渲染到 Streamlit 页面
    st.pyplot(plt.gcf())
    plt.close()
