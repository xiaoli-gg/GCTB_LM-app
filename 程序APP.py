import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('Bag_model.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "Surgy": {"type": "categorical", "options": [0, 1, 2, 3]},
#    "Age": {"type": "categorical", "min": 0.0, "max": 14417.0, "default": 5000.0},
    "Age": {"type": "categorical", "options": [0, 1]},
    "T": {"type": "categorical", "options": [0, 1, 2, 3, 4]},
    "N": {"type": "categorical", "options": [0, 1, 2]},
    "Tumor_Size": {"type": "categorical", "options": [0, 1, 2]},
}

# Streamlit 界面
#st.title("Prediction Model with SHAP Visualization")
st.title("Artificial intelligence-assisted Lung metastasis and prognosis model for patients with GCTB")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与解释
if st.button("Predict"):
    # 构造输入
    features = np.array([feature_values])
    feature_df = pd.DataFrame(features, columns=feature_ranges.keys())

    # 预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[1] * 100  # 二分类中通常 [1] 为“阳性”类

    # 显示预测结果
    text = f"Predicted probability of Lung Metastasis: {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(0.5, 0.5, text, fontsize=16, ha='center', va='center',
            fontname='Times New Roman', transform=ax.transAxes)
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # SHAP 分析（使用二分类结构）
    background = pd.read_csv("shap_background.csv")
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(feature_df)

    a = 0
    shap_fig = shap.plots.force(
        explainer.expected_value[1],  # 如果是二分类，取第1类的 expected_value 
        shap_values[1][0],            # 第1类对应的 SHAP 值
        feature_df.iloc[0],
        matplotlib=True,
        show=False
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")


