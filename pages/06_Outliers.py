"""
Nexus AI — Page 06: Outlier Detection & Treatment Studio
--------------------------------------------------------
Detect, inspect, and treat anomalous observations using IQR, Z-Score,
Winsorization, and Isolation Forest algorithms.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.session_state import init_session_state, get_active_data, set_active_data, get_dataset_telemetry
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_before_after, render_step_navigation, render_sidebar_status, dataset_guard, render_html
from utils.preprocessing import handle_outliers
from utils.visualization import plot_boxplot, plot_histogram


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    if not dataset_guard():
        return

    df = get_active_data()
    t = get_dataset_telemetry(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    render_hero(
        badge="● ANOMALY PROTOCOL",
        title="Outlier Detection & Conditioning",
        subtitle="Detect extreme anomalies and apply boundary clipping, statistical winsorization, or observation filtering."
    )

    if not num_cols:
        st.warning("⚠️ No numerical columns available in active dataset for outlier processing.")
        render_step_navigation(prev_page="pages/05_Missing_Values.py", next_page="pages/07_Encoding.py")
        return

    # Top KPI Metrics
    metrics = [
        {"icon": "🔢", "label": "Numerical Features", "value": f"{len(num_cols)}", "sub": "Eligible for outlier analysis"},
        {"icon": "🧬", "label": "Active Observations", "value": f"{t['rows']:,}", "sub": "Dataset length"},
        {"icon": "📊", "label": "Memory Footprint", "value": f"{t['memory_mb']:.2f} MB"},
        {"icon": "⚡", "label": "Outliers Status", "value": "Treated" if st.session_state.get("outliers_done") else "Ready", "sub": "Pipeline stage 06", "delta_type": "pos"},
    ]
    render_metric_grid(metrics)

    # Visual Outlier Inspection
    with st.container(border=True):
        render_section_header("01", "Visual Distribution & Anomaly Inspector")
        c_sel1, c_sel2 = st.columns(2)
        with c_sel1:
            selected_inspect_col = st.selectbox("Select Feature to Inspect:", num_cols, key="outlier_inspect_col")
        with c_sel2:
            cat_opts = ["None"] + df.select_dtypes(include=["object", "category"]).columns.tolist()
            color_group_col = st.selectbox("Optional Color Segment:", cat_opts, key="outlier_inspect_color")

        c_plot1, c_plot2 = st.columns(2)
        with c_plot1:
            st.plotly_chart(plot_boxplot(df, y_col=selected_inspect_col, color_col=color_group_col), use_container_width=True)
        with c_plot2:
            st.plotly_chart(plot_histogram(df, x_col=selected_inspect_col, color_col=color_group_col, kde=True), use_container_width=True)

    # Treatment Configurator
    render_html("<br>")
    with st.container(border=True):
        render_section_header("02", "Anomaly Treatment Configuration")

        col_conf1, col_conf2 = st.columns(2)
        with col_conf1:
            method = st.selectbox(
                "Select Detection Algorithm:",
                [
                    "IQR (Interquartile Range Rule)",
                    "Z-Score (Standard Deviations from Mean)",
                    "Winsorization (Percentile Boundary Capping)",
                    "Isolation Forest (Unsupervised ML Anomaly Detection)",
                ],
                key="outlier_method_select"
            )

            action = st.radio(
                "Treatment Action:",
                ["Clip / Cap Boundary Values (Preserve sample count)", "Remove Outlier Observations (Filter rows)"],
                index=0,
                key="outlier_action_radio"
            )
            action_val = "clip" if "Clip" in action else "remove"

        with col_conf2:
            target_cols = st.multiselect("Select Target Numerical Features:", num_cols, default=num_cols, key="outlier_target_cols")

            factor = 1.5
            lower_pct = 0.05
            upper_pct = 0.05
            contamination = 0.05

            if "IQR" in method:
                factor = st.slider("IQR Multiplier Threshold:", 1.0, 3.5, 1.5, step=0.25, help="1.5 is standard, 3.0 captures extreme outliers only.")
            elif "Z-Score" in method:
                factor = st.slider("Z-Score Multiplier (|z| > threshold):", 2.0, 4.5, 3.0, step=0.25)
            elif "Winsorization" in method:
                c_w1, c_w2 = st.columns(2)
                lower_pct = c_w1.slider("Lower Limit Percentile:", 0.01, 0.15, 0.05, step=0.01)
                upper_pct = c_w2.slider("Upper Limit Percentile:", 0.01, 0.15, 0.05, step=0.01)
            elif "Isolation" in method:
                contamination = st.slider("Expected Contamination Rate:", 0.01, 0.20, 0.05, step=0.01)

        if st.button("⚡ EXECUTE OUTLIER TREATMENT", use_container_width=True, key="btn_apply_outliers"):
            if not target_cols:
                st.error("Please select at least one feature.")
            else:
                method_code = "iqr"
                if "Z-Score" in method:
                    method_code = "zscore"
                elif "Winsorization" in method:
                    method_code = "winsorize"
                elif "Isolation" in method:
                    method_code = "isolation_forest"

                df_treated, count_outliers = handle_outliers(
                    df,
                    columns=target_cols,
                    method=method_code,
                    action=action_val,
                    factor=factor,
                    lower_pct=lower_pct,
                    upper_pct=upper_pct,
                    contamination=contamination,
                )

                st.session_state.outliers_done = True
                set_active_data(df_treated, stage_name="Outliers", log_entry=f"Treated outliers on {len(target_cols)} features using {method} (action: {action_val}). Affected points: {count_outliers:,}.")
                st.success(f"✓ Outlier treatment applied! ({count_outliers:,} points adjusted/removed)")
                st.rerun()

    # Before / After Matrix
    orig_df = st.session_state.get("original_data")
    if orig_df is not None:
        render_html("<br>")
        render_section_header("03", "Outlier Conditioning Audit")
        render_before_after(orig_df, df, note="Telemetry audit before and after outlier management.")

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/05_Missing_Values.py",
        next_page="pages/07_Encoding.py",
        prev_label="← Missing Values",
        next_label="Categorical Encoding Studio →"
    )


if __name__ == "__main__":
    main()
