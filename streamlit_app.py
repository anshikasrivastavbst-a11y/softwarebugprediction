"""
streamlit_app.py — Modern UI for the Software Bug Prediction System.

Visual layer redesign:
- Modern Enterprise Slate & Indigo design system
- Responsive card layouts, metric tiles, and organized analysis tabs
- Custom CSS for typography, borders, subtle shadows, and status badges
- 100% preservation of all underlying inference logic, inputs, and outputs
"""

import streamlit as st
import pandas as pd
from inference import predict_dataframe, get_feature_names, get_threshold

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Software Bug Prediction Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom Styling (Visual Layer Only)
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Global font & smoothing */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Header Container */
.hero-container {
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
    border-radius: 16px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 2rem;
    color: #FFFFFF;
    box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.hero-badge-container {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
}

.hero-badge {
    background: rgba(255, 255, 255, 0.18);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: #F8FAFC;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
    margin: 0 0 0.6rem 0;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.02rem;
    color: #E0E7FF;
    max-width: 820px;
    line-height: 1.55;
    margin: 0;
}

/* Metric KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 1.2rem;
    margin: 1.2rem 0;
}

.kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(15, 23, 42, 0.08);
}

.kpi-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}

.kpi-value {
    font-size: 1.95rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.1;
}

.kpi-footer {
    font-size: 0.8rem;
    margin-top: 0.45rem;
    color: #64748B;
}

.kpi-buggy {
    border-left: 4px solid #EF4444;
}

.kpi-buggy .kpi-value {
    color: #DC2626;
}

.kpi-clean {
    border-left: 4px solid #10B981;
}

.kpi-clean .kpi-value {
    color: #059669;
}

.kpi-total {
    border-left: 4px solid #4F46E5;
}

.kpi-total .kpi-value {
    color: #4338CA;
}

.kpi-rate {
    border-left: 4px solid #F59E0B;
}

.kpi-rate .kpi-value {
    color: #D97706;
}

/* Sidebar Model Panel */
.sidebar-panel {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.1rem;
    margin-bottom: 1.2rem;
}

.sidebar-panel-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

.sidebar-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px dashed #E2E8F0;
    font-size: 0.86rem;
}

.sidebar-stat-row:last-child {
    border-bottom: none;
}

.sidebar-stat-label {
    color: #64748B;
    font-weight: 500;
}

.sidebar-stat-val {
    color: #0F172A;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

.status-active-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: #059669;
    font-weight: 600;
    font-size: 0.78rem;
    background: #ECFDF5;
    padding: 0.2rem 0.55rem;
    border-radius: 9999px;
    border: 1px solid #A7F3D0;
}

.pulse-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #10B981;
    box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.3);
}

/* Feature Chip Tag */
.feature-chip {
    display: inline-block;
    background: #F1F5F9;
    color: #334155;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 0.2rem 0.45rem;
    margin: 0.15rem;
    border-radius: 6px;
    border: 1px solid #E2E8F0;
}

/* File Section Header Banner */
.dataset-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.85rem 1.25rem;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

.dataset-banner-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
}

.dataset-banner-meta {
    font-size: 0.82rem;
    color: #64748B;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 0.25rem 0.65rem;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}

/* Empty State Greeting Cards */
.empty-guide-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.4rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    height: 100%;
}

.empty-guide-icon {
    font-size: 1.8rem;
    margin-bottom: 0.65rem;
}

.empty-guide-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.4rem;
}

.empty-guide-desc {
    font-size: 0.88rem;
    color: #64748B;
    line-height: 1.5;
}

/* Download button container styling */
.download-action-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: #F1F5F9;
    border-radius: 10px;
    margin-bottom: 1rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar: System Specifications & Feature Explorer
# -----------------------------------------------------------------------------
features_list = get_feature_names()
threshold_value = get_threshold()

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-panel">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.6rem;">
                <span class="status-active-badge">
                    <span class="pulse-dot"></span> Model Ready
                </span>
                <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 500;">v2.0 Ensembled</span>
            </div>
            <div class="sidebar-panel-title">🛡️ System Specifications</div>
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">Decision Threshold</span>
                <span class="sidebar-stat-val">{threshold_value:.3f}</span>
            </div>
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">Active Features</span>
                <span class="sidebar-stat-val">{len(features_list)}</span>
            </div>
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">Base Learners</span>
                <span class="sidebar-stat-val">XGB + LGB + RF</span>
            </div>
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">Meta-Learner</span>
                <span class="sidebar-stat-val">LogisticRegression</span>
            </div>
            <div class="sidebar-stat-row">
                <span class="sidebar-stat-label">Calibration</span>
                <span class="sidebar-stat-val">Isotonic</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander(f"📋 Model Feature Lexicon ({len(features_list)} Features)", expanded=False):
        st.caption("CK Object-Oriented and Halstead Complexity metric schema:")
        chips_html = "".join([f'<span class="feature-chip">{f}</span>' for f in features_list])
        st.markdown(f'<div style="max-height: 240px; overflow-y: auto; padding-right: 4px;">{chips_html}</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ Quick Guidance", expanded=False):
        st.markdown(
            """
            **Supported Data Inputs:**
            - CSV exports containing code metrics (`wmc`, `dit`, `rfc`, `loc`, etc.)
            - Automatically harmonizes missing metrics by imputing with zero
            - Multiple dataset uploads evaluated concurrently
            """
        )

# -----------------------------------------------------------------------------
# Main Header / Hero Banner
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge-container">
            <span class="hero-badge">⚡ Stacking Architecture</span>
            <span class="hero-badge">🔬 Isotonic Calibrated</span>
            <span class="hero-badge">📊 PROMISE & NASA MDP Benchmarked</span>
        </div>
        <h1 class="hero-title">Software Defect Prediction Studio</h1>
        <p class="hero-subtitle">
            Automated software defect triage powered by an ensemble of XGBoost, LightGBM, and Random Forest 
            meta-learning with calibrated decision boundaries.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# File Upload Section
# -----------------------------------------------------------------------------
st.markdown("#### 📂 Select Module Datasets")
uploaded_files = st.file_uploader(
    label="Upload one or more CSV files containing module metrics for automated defect analysis",
    type="csv",
    accept_multiple_files=True,
    help="Supports standard benchmark datasets and custom CK / Halstead metric CSV exports."
)

# -----------------------------------------------------------------------------
# Prediction & Results Processing
# -----------------------------------------------------------------------------
if uploaded_files:
    for file in uploaded_files:
        # Exact data extraction
        original_df = pd.read_csv(file)
        predictions, probabilities = predict_dataframe(original_df)

        # Exact output dataframe construction
        result = original_df.copy()
        result["Bug Probability"] = probabilities.round(4)
        result["Prediction"] = pd.Series(predictions).map(
            {1: "🐞 Buggy", 0: "✅ Not Buggy"}
        ).values

        # Exact metric values
        total_count = len(predictions)
        buggy = int(predictions.sum())
        clean = int((predictions == 0).sum())
        defect_rate = (buggy / total_count * 100) if total_count > 0 else 0.0

        # Dataset Section Banner
        st.markdown(
            f"""
            <div class="dataset-banner">
                <div class="dataset-banner-title">
                    📄 {file.name}
                </div>
                <div class="dataset-banner-meta">
                    {total_count:,} modules • {len(original_df.columns)} input features
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Executive KPI Grid
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card kpi-total">
                    <div class="kpi-title">Total Evaluated</div>
                    <div class="kpi-value">{total_count:,}</div>
                    <div class="kpi-footer">Software modules analyzed</div>
                </div>
                <div class="kpi-card kpi-buggy">
                    <div class="kpi-title">🐞 Buggy Modules</div>
                    <div class="kpi-value">{buggy:,}</div>
                    <div class="kpi-footer">{defect_rate:.1f}% defect probability risk</div>
                </div>
                <div class="kpi-card kpi-clean">
                    <div class="kpi-title">✅ Clean Modules</div>
                    <div class="kpi-value">{clean:,}</div>
                    <div class="kpi-footer">{100 - defect_rate:.1f}% defect-free</div>
                </div>
                <div class="kpi-card kpi-rate">
                    <div class="kpi-title">Defect Ratio</div>
                    <div class="kpi-value">{defect_rate:.1f}%</div>
                    <div class="kpi-footer">Threshold: {threshold_value:.3f}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Organized Analysis Tabs
        tab_results, tab_risk, tab_raw = st.tabs([
            "🎯 Prediction Results",
            "📊 Defect Risk Summary",
            "📋 Original Data"
        ])

        with tab_results:
            csv = result.to_csv(index=False).encode("utf-8")
            
            # Action bar with export button
            col_info, col_btn = st.columns([3, 1])
            with col_info:
                st.caption(
                    "Each row reflects the model's calibrated probability and final binary defect classification."
                )
            with col_btn:
                st.download_button(
                    label="⬇ Download Prediction CSV",
                    data=csv,
                    file_name=f"prediction_{file.name}",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )

            st.dataframe(result, use_container_width=True, height=360)

        with tab_risk:
            st.markdown("##### Defect Class Distribution")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                summary_df = pd.DataFrame({
                    "Status": ["Buggy", "Not Buggy"],
                    "Count": [buggy, clean]
                }).set_index("Status")
                st.bar_chart(summary_df, color="#4F46E5", use_container_width=True)

            with chart_col2:
                st.markdown("##### Probability Calibration Bands")
                if len(probabilities) > 0:
                    prob_series = pd.Series(probabilities)
                    bands_df = pd.DataFrame({
                        "Risk Band": ["Low (<0.30)", "Medium (0.30 - 0.475)", "High (0.475 - 0.70)", "Critical (>=0.70)"],
                        "Count": [
                            int((prob_series < 0.30).sum()),
                            int(((prob_series >= 0.30) & (prob_series < threshold_value)).sum()),
                            int(((prob_series >= threshold_value) & (prob_series < 0.70)).sum()),
                            int((prob_series >= 0.70).sum())
                        ]
                    }).set_index("Risk Band")
                    st.bar_chart(bands_df, color="#F59E0B", use_container_width=True)

        with tab_raw:
            st.caption("Raw source data as loaded from the uploaded CSV prior to preprocessing.")
            st.dataframe(original_df, use_container_width=True, height=360)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

else:
    # Empty State: Visual guidance walkthrough
    st.markdown(
        """
        <div style="margin: 2rem 0;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1.2rem;">
                <div class="empty-guide-card">
                    <div class="empty-guide-icon">📂</div>
                    <div class="empty-guide-title">1. Upload Datasets</div>
                    <div class="empty-guide-desc">
                        Drop one or more metric CSV files into the upload zone above. Benchmarks from PROMISE or NASA MDP are supported out of the box.
                    </div>
                </div>
                <div class="empty-guide-card">
                    <div class="empty-guide-icon">⚡</div>
                    <div class="empty-guide-title">2. Ensemble Inference</div>
                    <div class="empty-guide-desc">
                        Inputs are preprocessed, scaled via RobustScaler, and classified with our calibrated multi-model stacking ensemble.
                    </div>
                </div>
                <div class="empty-guide-card">
                    <div class="empty-guide-icon">📥</div>
                    <div class="empty-guide-title">3. Inspect & Export</div>
                    <div class="empty-guide-desc">
                        Analyze defect density, sort modules by probability risk, and export timestamped CSV results for engineering triage.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.info("💡 Ready to begin: select or drop CSV file(s) into the upload area above to generate predictions.")
