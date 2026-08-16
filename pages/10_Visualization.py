"""
Nexus AI — Page 10: Interactive Visual Analytics & 3D Feature Space
-------------------------------------------------------------------
Interactive 2D visualization controls (histograms, box plots, scatter with trendlines,
correlation heatmaps, donut distributions) and 3D Feature Space Studio.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.session_state import init_session_state, get_active_data, get_dataset_telemetry
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_step_navigation, render_sidebar_status, dataset_guard, render_html
from utils.visualization import (
    plot_histogram,
    plot_boxplot,
    plot_scatter_2d,
    plot_scatter_3d,
    plot_correlation_heatmap,
    plot_categorical_distribution,
    plot_3d_pca_space,
)
from sklearn.decomposition import PCA


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    if not dataset_guard():
        return

    df = get_active_data()
    t = get_dataset_telemetry(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = df.columns.tolist()

    render_hero(
        badge="● VISUAL TELEMETRY PROTOCOL",
        title="Interactive Visual Analytics & 3D Studio",
        subtitle="Dynamic multidimensional exploration, distribution diagnostics, correlation patterns, and 3D manifold projection."
    )

    # Top KPI Metrics
    metrics = [
        {"icon": "📊", "label": "Features Matrix", "value": f"{t['cols']}", "sub": f"{len(num_cols)} Num | {len(cat_cols)} Cat"},
        {"icon": "🧬", "label": "Observation Space", "value": f"{t['rows']:,}", "sub": "Total data coordinates"},
        {"icon": "🌌", "label": "3D Dimension Studio", "value": "Active", "sub": "Plotly 3D ready", "delta_type": "pos"},
        {"icon": "⚡", "label": "Telemetry Status", "value": "Interactive", "sub": "Pipeline stage 10", "delta_type": "pos"},
    ]
    render_metric_grid(metrics)

    # Visual Analytics Tabs
    tab_2d, tab_3d, tab_corr = st.tabs([
        "📊 2D Interactive Analytics Studio",
        "🌌 3D Feature Space & PCA Studio",
        "🔥 Correlation Heatmap Matrix",
    ])

    # 1. 2D Studio
    with tab_2d:
        render_section_header("01", "2D Chart Generator & Dynamic Controls")
        
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1, 1, 1])
        with c_ctrl1:
            chart_type = st.selectbox(
                "Chart Type:",
                ["Histogram / Distribution", "2D Scatter Plot", "Box / Outlier Plot", "Categorical Frequency (Bar/Donut)"],
                key="vis_2d_chart_type"
            )

        with c_ctrl2:
            if chart_type == "Histogram / Distribution":
                x_axis = st.selectbox("Select Feature (X-Axis):", num_cols if num_cols else all_cols, key="vis_hist_x")
                color_opt = ["None"] + all_cols
                color_axis = st.selectbox("Color / Hue Segment:", color_opt, key="vis_hist_color")
            elif chart_type == "2D Scatter Plot":
                x_axis = st.selectbox("X-Axis Feature:", num_cols if num_cols else all_cols, key="vis_scat_x")
                y_axis = st.selectbox("Y-Axis Feature:", [c for c in num_cols if c != x_axis] if len(num_cols) > 1 else all_cols, key="vis_scat_y")
            elif chart_type == "Box / Outlier Plot":
                y_axis = st.selectbox("Numerical Feature (Y-Axis):", num_cols if num_cols else all_cols, key="vis_box_y")
                cat_opts = ["None"] + cat_cols
                x_axis = st.selectbox("Grouping Feature (Optional X-Axis):", cat_opts, key="vis_box_x")
            elif chart_type == "Categorical Frequency (Bar/Donut)":
                cat_target = st.selectbox("Categorical Feature:", cat_cols if cat_cols else all_cols, key="vis_cat_target")
                sub_style = st.radio("Style:", ["Bar Chart", "Donut Chart"], horizontal=True, key="vis_cat_style")

        with c_ctrl3:
            if chart_type == "Histogram / Distribution":
                n_bins = st.slider("Number of Histogram Bins:", 5, 100, 30)
                add_kde = st.checkbox("Show Marginal Box Plot", value=True)
            elif chart_type == "2D Scatter Plot":
                color_opt = ["None"] + all_cols
                color_axis = st.selectbox("Color Grouping:", color_opt, key="vis_scat_color")
                add_ols = st.checkbox("Add Linear OLS Trendline", value=False)
            elif chart_type == "Box / Outlier Plot":
                color_opt = ["None"] + all_cols
                color_axis = st.selectbox("Color Segment:", color_opt, key="vis_box_color")

        # Render Chart
        render_html("<br>")
        if chart_type == "Histogram / Distribution":
            fig = plot_histogram(df, x_col=x_axis, color_col=color_axis, n_bins=n_bins, kde=add_kde)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "2D Scatter Plot":
            fig = plot_scatter_2d(df, x_col=x_axis, y_col=y_axis, color_col=color_axis, trendline=add_ols)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Box / Outlier Plot":
            fig = plot_boxplot(df, y_col=y_axis, x_col=x_axis, color_col=color_axis)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type == "Categorical Frequency (Bar/Donut)":
            fig = plot_categorical_distribution(df, col=cat_target, chart_type="donut" if "Donut" in sub_style else "bar")
            st.plotly_chart(fig, use_container_width=True)

    # 2. 3D Studio
    with tab_3d:
        render_section_header("02", "3D Interactive Feature Space Studio")
        if len(num_cols) < 3:
            st.warning("⚠️ 3D feature space visualization requires at least 3 numerical features.")
        else:
            mode_3d = st.radio("3D Mode:", ["Raw 3-Feature Coordinates", "PCA 3D Manifold Projection Space"], horizontal=True)

            if mode_3d == "Raw 3-Feature Coordinates":
                c3_1, c3_2, c3_3, c3_4 = st.columns(4)
                with c3_1:
                    x_3d = st.selectbox("X-Axis (3D):", num_cols, index=0, key="vis_3d_x")
                with c3_2:
                    y_3d = st.selectbox("Y-Axis (3D):", num_cols, index=min(1, len(num_cols)-1), key="vis_3d_y")
                with c3_3:
                    z_3d = st.selectbox("Z-Axis (3D):", num_cols, index=min(2, len(num_cols)-1), key="vis_3d_z")
                with c3_4:
                    color_3d_opts = ["None"] + all_cols
                    c_3d = st.selectbox("Color Segment (3D):", color_3d_opts, index=0, key="vis_3d_c")

                fig_3d = plot_scatter_3d(df, x_col=x_3d, y_col=y_3d, z_col=z_3d, color_col=c_3d)
                st.plotly_chart(fig_3d, use_container_width=True)

            else:
                st.markdown("#### 🌌 PCA 3D Feature Space Projection")
                pca = PCA(n_components=3)
                clean_num = df[num_cols].dropna()
                if len(clean_num) > 10:
                    pca_res = pca.fit_transform(clean_num)
                    pca_3d_df = pd.DataFrame({
                        "PC_1": pca_res[:, 0],
                        "PC_2": pca_res[:, 1],
                        "PC_3": pca_res[:, 2],
                    }, index=clean_num.index)

                    color_pca_opts = ["None"] + all_cols
                    pca_col_sel = st.selectbox("Color Data Points By:", color_pca_opts, index=0, key="vis_3d_pca_color")
                    if pca_col_sel != "None":
                        pca_3d_df[pca_col_sel] = df.loc[clean_num.index, pca_col_sel].astype(str)

                    fig_pca_3d = plot_3d_pca_space(pca_3d_df, color_col=pca_col_sel if pca_col_sel != "None" else None, explained_variance=pca.explained_variance_ratio_)
                    st.plotly_chart(fig_pca_3d, use_container_width=True)

                    render_html(
                        f"""
                        <div style="display:flex; gap:15px; font-size:13px; color:#38BDF8; font-weight:700;">
                            <span>PC1: {pca.explained_variance_ratio_[0]*100:.1f}% Variance</span>
                            <span>PC2: {pca.explained_variance_ratio_[1]*100:.1f}% Variance</span>
                            <span>PC3: {pca.explained_variance_ratio_[2]*100:.1f}% Variance</span>
                            <span style="color:#10B981;">Total Captured: {pca.explained_variance_ratio_.sum()*100:.1f}%</span>
                        </div>
                        """
                    )

    # 3. Correlation Heatmap
    with tab_corr:
        render_section_header("03", "Full Pearson Correlation Matrix")
        if len(num_cols) >= 2:
            st.plotly_chart(plot_correlation_heatmap(df, num_cols), use_container_width=True)
        else:
            st.info("ℹ️ Correlation matrix requires at least 2 numerical features.")

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/09_Scaling.py",
        next_page="pages/11_Modeling.py",
        prev_label="← Feature Scaling",
        next_label="Model Selection & Training Studio →"
    )


if __name__ == "__main__":
    main()
