"""
========================================================================================
⚡ NEXUS AI — Autonomous Machine Learning & Data Science Platform
========================================================================================
Main application entry point orchestrating end-to-end multi-page navigation,
global session state synchronization, and cyberpunk glassmorphism layout.
"""

import streamlit as st
from utils.session_state import init_session_state
from utils.ui import inject_global_css

# ============================================================
# GLOBAL APP CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Nexus AI — Machine Learning Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Global Shared State & Inject Global CSS
init_session_state()
inject_global_css()

# ============================================================
# STRUCTURED WORKFLOW NAVIGATION HIERARCHY
# ============================================================
pages = {
    "🚀 Platform Hub": [
        st.Page("pages/01_Dashboard.py", title="Executive Dashboard", icon="📊", default=True),
        st.Page("pages/02_Upload.py", title="Dataset Ingestion", icon="📥"),
        st.Page("pages/03_Data_Overview.py", title="Data Overview", icon="🔍"),
    ],
    "🧹 Preprocessing Lab": [
        st.Page("pages/04_Cleaning.py", title="Data Cleaning & Duplicates", icon="🧹"),
        st.Page("pages/05_Missing_Values.py", title="Missing Values Lab", icon="🩹"),
        st.Page("pages/06_Outliers.py", title="Outlier Detection", icon="⚡"),
    ],
    "⚙️ Feature Engineering": [
        st.Page("pages/07_Encoding.py", title="Categorical Encoding", icon="🔤"),
        st.Page("pages/08_Feature_Engineering.py", title="Feature Engineering & PCA", icon="🧬"),
        st.Page("pages/09_Scaling.py", title="Feature Scaling", icon="📏"),
    ],
    "📊 Visual Intelligence": [
        st.Page("pages/10_Visualization.py", title="Visual Analytics & 3D", icon="🌌"),
    ],
    "🧠 Machine Learning": [
        st.Page("pages/11_Modeling.py", title="Model Studio & Training", icon="🤖"),
        st.Page("pages/12_Evaluation.py", title="Model Diagnostics & Eval", icon="🎯"),
    ],
    "💾 Artifacts & Reports": [
        st.Page("pages/13_Export.py", title="Export Center & Report", icon="💾"),
    ],
}

# Orchestrate and execute active page
pg = st.navigation(pages)
pg.run()
