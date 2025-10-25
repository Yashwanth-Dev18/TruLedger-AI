import streamlit as st
import os
import json
import pandas as pd
from dataProcessor import process_transaction_data
from ML_Engine import run_fraud_detection

st.set_page_config(page_title="TruLedger - AI Fraud Detection", layout="wide")

# ===============================
# 🏦 HEADER
# ===============================
st.markdown('<h1 style="text-align:center;color:#1f77b4;">🏦 TruLedger</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#666;">Explainable AI Prototype for ML Fraud Detection</p>', unsafe_allow_html=True)

# ===============================
# 📁 Dataset Selection
# ===============================
dataset_options = {
    "TransactionLogs-1": "1,000 records",
    "TransactionLogs-2": "5,000 records",
    "TransactionLogs-3": "10,000 records"
}
selected_dataset = st.selectbox("Choose a dataset:", list(dataset_options.keys()))
st.info(f"**{selected_dataset}**: {dataset_options[selected_dataset]}")

if st.button("🚀 Run Fraud Detection"):
    input_file = os.path.join("Uploaded_Datasets", "Raw", f"{selected_dataset}.csv")
    processed_file = process_transaction_data(input_file)
    if processed_file is None:
        st.error("❌ Failed to process data.")
    else:
        with st.spinner("Running fraud detection..."):
            success = run_fraud_detection(processed_file)
        if success:
            st.success("✅ Fraud detection complete!")
            fraud_df = pd.read_csv('detected_fraud_transactions.csv')
            st.dataframe(fraud_df.head(10), use_container_width=True)
        else:
            st.error("❌ Fraud detection failed.")

# ===============================
# 📊 Visualization
# ===============================
def load_viz_data():
    try:
        with open('app_visualization_data.json', 'r') as f:
            return json.load(f)
    except:
        return None

viz_data = load_viz_data()
if viz_data:
    st.header("📊 Fraud Pattern Analysis")
    st.subheader("Top Job Categories")
    st.bar_chart(pd.Series(viz_data['job_categories']))
    st.subheader("Top Transaction Categories")
    st.bar_chart(pd.Series(viz_data['transaction_categories']))
    st.subheader("Age Groups")
    st.bar_chart(pd.Series(viz_data['age_groups']))
    st.subheader("Amount Analysis")
    st.metric("Fraud Avg", f"${viz_data['amount_analysis']['fraud_avg']:.2f}")
    st.metric("Normal Avg", f"${viz_data['amount_analysis']['normal_avg']:.2f}")
    st.metric("Increase", f"{viz_data['amount_analysis']['increase_pct']:.1f}%")
