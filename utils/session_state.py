"""
Nexus AI — Session State Management & Pipeline Tracker
------------------------------------------------------
Provides persistent, reactive state synchronization across all 13 workflow stages.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets


def init_session_state():
    """Initialize all global session state keys with robust defaults."""
    defaults = {
        # Dataset Core
        "data": None,
        "original_data": None,
        "file_name": None,
        "file_key": None,
        "dataset_name": "No dataset loaded",

        # Preprocessing Flags
        "duplicates_done": False,
        "missing_done": False,
        "outliers_done": False,
        "cleaning_done": False,
        "encoding_done": False,
        "feature_eng_done": False,
        "scaling_done": False,
        "imbalance_done": False,
        "feature_selection_done": False,

        # Transformation Audits & Logs
        "transformation_log": [],
        "cleaning_log": [],
        "encoding_log": [],
        "scaling_log": [],
        "feature_eng_log": [],

        # Feature Metadata
        "original_numeric_columns": [],
        "original_categorical_columns": [],
        "selected_features": [],
        "target_column": None,
        "problem_type": "Classification",

        # Machine Learning State
        "model": None,
        "model_name": None,
        "training_time": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "y_pred": None,
        "y_prob": None,
        "cluster_labels": None,
        "evaluation_metrics": {},
        "feature_importance_df": None,
        "predictions_df": None,
        "confusion_matrix": None,

        # Navigation & History
        "current_page": "01_Dashboard",
        "stage_history": [],
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_active_data() -> pd.DataFrame:
    """Return the current working dataset from session state or None."""
    return st.session_state.get("data")


def set_active_data(df: pd.DataFrame, stage_name: str = None, log_entry: str = None):
    """Update active working data and record in transformation audit log."""
    st.session_state.data = df.copy()
    if log_entry:
        if "transformation_log" not in st.session_state or not isinstance(st.session_state.transformation_log, list):
            st.session_state.transformation_log = []
        st.session_state.transformation_log.append(f"[{stage_name or 'Pipeline'}] {log_entry}")


def get_dataset_telemetry(df: pd.DataFrame = None) -> dict:
    """Compute high-level telemetry stats for a dataframe."""
    if df is None:
        df = st.session_state.get("data")
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "rows": 0,
            "cols": 0,
            "numeric_count": 0,
            "categorical_count": 0,
            "missing_count": 0,
            "missing_pct": 0.0,
            "duplicate_count": 0,
            "duplicate_pct": 0.0,
            "memory_mb": 0.0,
            "health_score": 0,
        }

    rows, cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    missing_count = int(df.isnull().sum().sum())
    total_cells = rows * cols if (rows * cols) > 0 else 1
    missing_pct = (missing_count / total_cells) * 100
    duplicate_count = int(df.duplicated().sum())
    duplicate_pct = (duplicate_count / rows * 100) if rows > 0 else 0.0
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    # Simple Health Score from 0 to 100
    score = 100
    if missing_pct > 0:
        score -= min(35, int(missing_pct * 1.5))
    if duplicate_pct > 0:
        score -= min(25, int(duplicate_pct * 1.2))
    if cols < 2:
        score -= 20
    health_score = max(10, min(100, score))

    return {
        "rows": rows,
        "cols": cols,
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "duplicate_count": duplicate_count,
        "duplicate_pct": duplicate_pct,
        "memory_mb": memory_mb,
        "health_score": health_score,
    }


def get_workflow_status() -> list:
    """Return ordered pipeline stages and their completion status."""
    has_data = st.session_state.get("data") is not None
    stages = [
        {
            "id": "upload",
            "name": "Dataset Ingestion",
            "page": "pages/02_Upload.py",
            "icon": "📥",
            "done": has_data,
        },
        {
            "id": "overview",
            "name": "Data Overview",
            "page": "pages/03_Data_Overview.py",
            "icon": "🔍",
            "done": has_data,
        },
        {
            "id": "cleaning",
            "name": "Data Cleaning",
            "page": "pages/04_Cleaning.py",
            "icon": "🧹",
            "done": st.session_state.get("cleaning_done", False) or st.session_state.get("duplicates_done", False),
        },
        {
            "id": "missing",
            "name": "Missing Values",
            "page": "pages/05_Missing_Values.py",
            "icon": "🩹",
            "done": st.session_state.get("missing_done", False),
        },
        {
            "id": "outliers",
            "name": "Outliers Treatment",
            "page": "pages/06_Outliers.py",
            "icon": "⚡",
            "done": st.session_state.get("outliers_done", False),
        },
        {
            "id": "encoding",
            "name": "Categorical Encoding",
            "page": "pages/07_Encoding.py",
            "icon": "🔤",
            "done": st.session_state.get("encoding_done", False),
        },
        {
            "id": "feature_eng",
            "name": "Feature Engineering",
            "page": "pages/08_Feature_Engineering.py",
            "icon": "🧬",
            "done": st.session_state.get("feature_eng_done", False),
        },
        {
            "id": "scaling",
            "name": "Feature Scaling",
            "page": "pages/09_Scaling.py",
            "icon": "📏",
            "done": st.session_state.get("scaling_done", False),
        },
        {
            "id": "visualization",
            "name": "Visual Analytics",
            "page": "pages/10_Visualization.py",
            "icon": "📊",
            "done": has_data,
        },
        {
            "id": "modeling",
            "name": "Model Training",
            "page": "pages/11_Modeling.py",
            "icon": "🧠",
            "done": st.session_state.get("model") is not None,
        },
        {
            "id": "evaluation",
            "name": "Model Evaluation",
            "page": "pages/12_Evaluation.py",
            "icon": "🎯",
            "done": bool(st.session_state.get("evaluation_metrics")),
        },
        {
            "id": "export",
            "name": "Export Center",
            "page": "pages/13_Export.py",
            "icon": "💾",
            "done": False,
        },
    ]
    return stages


def reset_workflow(keep_original_file: bool = True):
    """Cleanly reset pipeline state."""
    reset_keys = [
        "duplicates_done", "missing_done", "outliers_done", "cleaning_done",
        "encoding_done", "feature_eng_done", "scaling_done", "imbalance_done",
        "feature_selection_done", "transformation_log", "cleaning_log",
        "encoding_log", "scaling_log", "feature_eng_log", "selected_features",
        "target_column", "model", "model_name", "training_time", "X_train",
        "X_test", "y_train", "y_test", "y_pred", "y_prob", "cluster_labels",
        "evaluation_metrics", "feature_importance_df", "predictions_df", "confusion_matrix"
    ]

    for key in reset_keys:
        if isinstance(st.session_state.get(key), list):
            st.session_state[key] = []
        elif isinstance(st.session_state.get(key), dict):
            st.session_state[key] = {}
        elif isinstance(st.session_state.get(key), bool):
            st.session_state[key] = False
        else:
            st.session_state[key] = None

    if keep_original_file and st.session_state.get("original_data") is not None:
        st.session_state.data = st.session_state.original_data.copy()
        st.session_state.original_numeric_columns = (
            st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
        )
        st.session_state.original_categorical_columns = (
            st.session_state.data.select_dtypes(include=["object", "category"]).columns.tolist()
        )
    elif not keep_original_file:
        st.session_state.data = None
        st.session_state.original_data = None
        st.session_state.file_name = None
        st.session_state.file_key = None
        st.session_state.dataset_name = "No dataset loaded"


def load_sample_dataset(name: str) -> pd.DataFrame:
    """Load a standard built-in dataset for instant demonstration."""
    name_lower = name.lower()
    df = None
    if "iris" in name_lower:
        raw = datasets.load_iris(as_frame=True)
        df = raw.frame.rename(columns={"target": "species"})
        # Map species targets to string names
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        df["species"] = df["species"].map(species_map)
    elif "wine" in name_lower:
        raw = datasets.load_wine(as_frame=True)
        df = raw.frame.rename(columns={"target": "wine_class"})
    elif "california" in name_lower or "housing" in name_lower:
        raw = datasets.fetch_california_housing(as_frame=True)
        df = raw.frame.rename(columns={"MedHouseVal": "HousePrice"})
    elif "breast" in name_lower or "cancer" in name_lower:
        raw = datasets.load_breast_cancer(as_frame=True)
        df = raw.frame.rename(columns={"target": "diagnosis"})
    elif "titanic" in name_lower:
        # Fallback synthetic / standard titanic if available
        try:
            url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
            df = pd.read_csv(url)
        except Exception:
            raw = datasets.load_iris(as_frame=True)
            df = raw.frame

    if df is not None:
        st.session_state.original_data = df.copy()
        st.session_state.data = df.copy()
        st.session_state.file_name = f"Sample: {name.title()}"
        st.session_state.dataset_name = name.title()
        st.session_state.file_key = (f"sample_{name_lower}", len(df))
        reset_workflow(keep_original_file=True)

    return df
