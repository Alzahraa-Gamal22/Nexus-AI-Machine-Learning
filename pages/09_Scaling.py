"""
Nexus AI — Page 09: Feature Normalization & Scaling Studio
----------------------------------------------------------
Standardize feature magnitudes using StandardScaler, MinMaxScaler,
RobustScaler, MaxAbsScaler, or Normalizer.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.session_state import init_session_state, get_active_data, set_active_data, get_dataset_telemetry
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_before_after, render_step_navigation, render_sidebar_status, dataset_guard, render_html
from utils.preprocessing import apply_feature_scaling
from utils.visualization import plot_histogram


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
        badge="● NORMALIZATION PROTOCOL",
        title="Feature Scaling & Standardization",
        subtitle="Align feature magnitudes to eliminate variance bias and accelerate machine learning convergence."
    )

    if not num_cols:
        st.warning("⚠️ No numerical columns available in active dataset for scaling.")
        render_step_navigation(prev_page="pages/08_Feature_Engineering.py", next_page="pages/10_Visualization.py")
        return

    # Intelligent feature filter (avoid scaling binary dummy features by default if desired)
    candidate_cols = [c for c in num_cols if df[c].nunique() > 2]
    if not candidate_cols:
        candidate_cols = num_cols

    # Top KPI Metrics
    metrics = [
        {"icon": "📏", "label": "Candidate Features", "value": f"{len(candidate_cols)}", "sub": f"Out of {len(num_cols)} numerical"},
        {"icon": "🧬", "label": "Observations", "value": f"{t['rows']:,}", "sub": "Dataset records"},
        {"icon": "📊", "label": "Memory Footprint", "value": f"{t['memory_mb']:.2f} MB"},
        {"icon": "⚡", "label": "Scaling Status", "value": "Scaled" if st.session_state.get("scaling_done") else "Ready", "sub": "Pipeline stage 09", "delta_type": "pos"},
    ]
    render_metric_grid(metrics)

    # Scaling Configurator
    with st.container(border=True):
        render_section_header("01", "Scaler Architecture Selection")
        c1, c2 = st.columns(2)

        with c1:
            scaler_choice = st.selectbox(
                "Select Normalization Algorithm:",
                [
                    "StandardScaler (Mean = 0, Std = 1 — Universal ML Standard)",
                    "MinMaxScaler (Compress values into [0, 1] range)",
                    "RobustScaler (Median & IQR — Immune to heavy outliers)",
                    "MaxAbsScaler (Scale by absolute maximum into [-1, 1])",
                    "Normalizer (Vector unit norm per observation)",
                ],
                key="scaler_select"
            )

            scaler_code = "standard"
            if "MinMax" in scaler_choice:
                scaler_code = "minmax"
            elif "Robust" in scaler_choice:
                scaler_code = "robust"
            elif "MaxAbs" in scaler_choice:
                scaler_code = "maxabs"
            elif "Normalizer" in scaler_choice:
                scaler_code = "normalizer"

        with c2:
            target_scale_cols = st.multiselect(
                "Select Features to Scale:",
                num_cols,
                default=candidate_cols,
                key="scale_target_features"
            )

        if st.button("⚡ EXECUTE FEATURE SCALING", use_container_width=True, key="btn_apply_scaling"):
            if not target_scale_cols:
                st.error("Please select at least one feature to scale.")
            else:
                df_scaled, scaler_obj = apply_feature_scaling(df, target_scale_cols, scaler_type=scaler_code)
                st.session_state.scaling_done = True
                if scaler_obj is not None and st.session_state.get("preprocessing_pipeline"):
                    st.session_state.preprocessing_pipeline.record_scaling(scaler_obj, scaler_code, target_scale_cols)
                set_active_data(df_scaled, stage_name="Scaling", log_entry=f"Applied {scaler_choice} on {len(target_scale_cols)} features: {', '.join(target_scale_cols)}")
                st.success(f"✓ Successfully scaled {len(target_scale_cols)} features using {scaler_code}!")
                st.rerun()

    # Visual Inspection Before / After
    if target_scale_cols:
        render_html("<br>")
        with st.container(border=True):
            render_section_header("02", "Post-Scaling Distribution Inspector")
            inspect_feat = st.selectbox("Inspect Feature Distribution:", target_scale_cols, key="scale_inspect_feat")
            st.plotly_chart(plot_histogram(df, x_col=inspect_feat, kde=True), use_container_width=True)

    # Before / After Matrix
    orig_df = st.session_state.get("original_data")
    if orig_df is not None:
        render_html("<br>")
        render_section_header("03", "Scaling Audit Telemetry")
        render_before_after(orig_df, df, note="Comparing raw magnitudes against standardized features.")

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/08_Feature_Engineering.py",
        next_page="pages/10_Visualization.py",
        prev_label="← Feature Engineering",
        next_label="Interactive Visual Analytics →"
    )


if __name__ == "__main__":
    main()
