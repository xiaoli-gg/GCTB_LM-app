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

import streamlit.components.v1 as components

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 构造输入 DataFrame
    features = np.array([feature_values])
    feature_df = pd.DataFrame(features, columns=feature_ranges.keys())

    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果图像
    text = f"Based on feature values, predicted possibility of Lung Metastasis is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 加载背景数据
    background = pd.read_csv("shap_background.csv")

    # 初始化 SHAP 解释器
    explainer = shap.KernelExplainer(model.predict_proba, background)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(feature_df)

    # 显示 SHAP 力图（嵌入 HTML 渲染）
    # shap.initjs()
    force_plot_html = shap.force_plot(
        explainer.expected_value[predicted_class],
        shap_values[predicted_class][0],
        feature_df.iloc[0],
        matplotlib=False,
        show=False
    )
    components.html(force_plot_html.html(), height=300)
