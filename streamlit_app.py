"""
streamlit_app.py — Streamlit UI for the Software Bug Prediction System.
Uses the stacking ensemble via the shared inference module.
"""

import streamlit as st
import pandas as pd
from inference import predict_dataframe, get_feature_names, get_threshold

st.set_page_config(
    page_title="Software Bug Prediction",
    page_icon="🐞",
    layout="wide"
)

st.title("🐞 Software Bug Prediction System")
st.caption(
    "Powered by a stacking ensemble: XGBoost + LightGBM + RandomForest → "
    "LogisticRegression (isotonic calibrated)"
)

with st.sidebar:
    st.header("Model Info")
    st.metric("Features used", len(get_feature_names()))
    st.metric("Decision threshold", f"{get_threshold():.3f}")
    st.write("**Features:**")
    st.write(get_feature_names())

st.write("Upload one or more CSV files to predict software defects.")

uploaded_files = st.file_uploader(
    "Upload CSV file(s)",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        st.divider()
        st.subheader(f"📄 {file.name}")

        original_df = pd.read_csv(file)
        st.write("### Original Data")
        st.dataframe(original_df, use_container_width=True)

        predictions, probabilities = predict_dataframe(original_df)

        result = original_df.copy()
        result["Bug Probability"] = probabilities.round(4)
        result["Prediction"] = pd.Series(predictions).map(
            {1: "🐞 Buggy", 0: "✅ Not Buggy"}
        ).values

        st.write("### Prediction Results")
        st.dataframe(result, use_container_width=True)

        csv = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Prediction CSV",
            data=csv,
            file_name=f"prediction_{file.name}",
            mime="text/csv"
        )

        buggy = int(predictions.sum())
        clean = int((predictions == 0).sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(predictions))
        col2.metric("🐞 Buggy", buggy)
        col3.metric("✅ Not Buggy", clean)
else:
    st.info("Upload one or more CSV files to begin.")
