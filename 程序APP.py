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

# 设置标题
st.set_page_config(page_title='AI-assisted Lung Metastasis Prediction Tool')
st.title("Artificial intelligence-assisted Lung metastasis and prognosis model for patients with GCTB")

# 定义特征范围
feature_ranges = {
    "Surgy": {"type": "categorical", "options": [0, 1, 2, 3]},
    "Age": {"type": "categorical", "options": [0, 1]},
    "T": {"type": "categorical", "options": [0, 1, 2, 3, 4]},
    "N": {"type": "categorical", "options": [0, 1, 2]},
    "Tumor_Size": {"type": "categorical", "options": [0, 1, 2]},
}

# 左侧输入面板
st.sidebar.header("🔢 Input Features")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 页面底部版权信息
st.sidebar.markdown("---")
st.sidebar.markdown("##### All rights reserved") 
st.sidebar.markdown("##### Contact: mengpanli163@163.com (Mengpan Li, Shanghai Jiao Tong University School of Medicine)")


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

    # SHAP 分析，注意“这里所用的模型是Bagclassifier，所以需要background数据，如果是tree模型的话就不需要”
    background = pd.read_csv("shap_background.csv")
    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(feature_df)

    # 因为feature_df只有一个样本，所以a只能为0
    a = 0
    shap_fig = shap.plots.force(
        explainer.expected_value,  
        shap_values[a], 
        feature_df.iloc[a, :],
        matplotlib=True,
        show=False
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")


