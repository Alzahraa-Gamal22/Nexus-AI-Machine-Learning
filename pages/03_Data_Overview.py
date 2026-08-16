"""
Nexus AI — Page 03: Exploratory Data Overview & Statistics
----------------------------------------------------------
Comprehensive statistical telemetry, distribution profiling, cardinality inspection,
and data health audit.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.session_state import init_session_state, get_active_data, get_dataset_telemetry
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_step_navigation, render_sidebar_status, dataset_guard


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    # Guard check
    if not dataset_guard():
        return

    df = get_active_data()
    t = get_dataset_telemetry(df)

    render_hero(
        badge="● EXPLORATORY AUDIT",
        title="Dataset Deep Inspection & Profiling",
        subtitle="Deep statistical profiling, feature cardinality introspection, distribution characteristics, and quality alerts."
    )

    # Top KPI Metrics
    metrics = [
        {"icon": "🧬", "label": "Total Samples", "value": f"{t['rows']:,}", "sub": f"{t['memory_mb']:.2f} MB memory"},
        {"icon": "🔢", "label": "Numerical", "value": f"{t['numeric_count']}", "sub": "Continuous & discrete"},
        {"icon": "🏷️", "label": "Categorical", "value": f"{t['categorical_count']}", "sub": "Text & categories"},
        {"icon": "🩺", "label": "Health Index", "value": f"{t['health_score']}/100", "sub": "Data readiness rating", "delta_type": "pos" if t['health_score'] >= 80 else "neg"},
    ]
    render_metric_grid(metrics)

    # Statistical Profiles
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    tab_num, tab_cat, tab_quality = st.tabs(["🔢 Numerical Profiles", "🏷️ Categorical Profiles", "🩺 Data Quality Checklist"])

    with tab_num:
        if num_cols:
            render_section_header("01", "Descriptive Statistics (Numerical)")
            desc_df = df[num_cols].describe().T
            desc_df["skewness"] = df[num_cols].skew()
            desc_df["kurtosis"] = df[num_cols].kurt()
            desc_df = desc_df.round(3)
            st.dataframe(desc_df, use_container_width=True)
        else:
            st.info("ℹ️ No numerical columns detected in this dataset.")

    with tab_cat:
        if cat_cols:
            render_section_header("02", "Categorical Profiles & Cardinality")
            cat_data = []
            for col in cat_cols:
                n_uniq = int(df[col].nunique(dropna=False))
                top_val = df[col].mode().iloc[0] if not df[col].dropna().empty else "N/A"
                top_freq = int(df[col].value_counts().iloc[0]) if not df[col].dropna().empty else 0
                pct_top = (top_freq / len(df)) * 100
                cat_data.append({
                    "Feature": col,
                    "Distinct Classes": n_uniq,
                    "Dominant Class": str(top_val),
                    "Dominant Frequency": f"{top_freq:,} ({pct_top:.1f}%)",
                    "Cardinality": "High (>30)" if n_uniq > 30 else ("Binary" if n_uniq == 2 else "Low/Med"),
                })
            cat_df = pd.DataFrame(cat_data)
            st.dataframe(cat_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No categorical columns detected in this dataset.")

    with tab_quality:
        render_section_header("03", "Data Quality & Anomaly Checklist")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 🩹 Missing Values Breakdown")
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            if not null_cols.empty:
                null_df = pd.DataFrame({
                    "Feature": null_cols.index,
                    "Missing Count": null_cols.values,
                    "Missing %": (null_cols.values / len(df) * 100).round(2)
                }).sort_values("Missing Count", ascending=False)
                st.dataframe(null_df, use_container_width=True, hide_index=True)
            else:
                st.success("✓ Zero missing cells detected across all columns.")

        with c2:
            st.markdown("#### 🔄 Duplicate & Constant Features")
            constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
            if constant_cols:
                st.warning(f"⚠️ Constant features (zero variance): {', '.join(constant_cols)}")
            else:
                st.success("✓ No zero-variance constant features detected.")

            if t["duplicate_count"] > 0:
                st.warning(f"⚠️ Found {t['duplicate_count']:,} duplicate rows ({t['duplicate_pct']:.1f}%). Proceed to Data Cleaning to handle them.")
            else:
                st.success("✓ No identical duplicate rows found.")

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/02_Upload.py",
        next_page="pages/04_Cleaning.py",
        prev_label="← Dataset Ingestion",
        next_label="Clean Dataset & Duplicates →"
    )


if __name__ == "__main__":
    main()
