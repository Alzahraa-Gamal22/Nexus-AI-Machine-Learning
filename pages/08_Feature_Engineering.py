"""
Nexus AI — Page 08: Feature Engineering & Dimensionality Reduction
------------------------------------------------------------------
Mathematical transforms (Log, Box-Cox, Yeo-Johnson), Polynomial Features,
Principal Component Analysis (PCA), Recursive Feature Elimination (RFE), and SMOTE.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.session_state import init_session_state, get_active_data, set_active_data, get_dataset_telemetry
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_before_after, render_step_navigation, render_sidebar_status, dataset_guard, render_html
from utils.preprocessing import transform_features, generate_polynomial_features, apply_pca, apply_rfe_selection, handle_imbalanced_classes


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
        badge="● TRANSFORMATION PROTOCOL",
        title="Feature Engineering & Dimensionality",
        subtitle="Apply nonlinear feature conditioning, polynomial expansion, PCA projection, and RFE feature selection."
    )

    # Top KPI Metrics
    selected_feats = st.session_state.get("selected_features", [])
    metrics = [
        {"icon": "📊", "label": "Active Features", "value": f"{t['cols']}", "sub": f"{len(num_cols)} numerical"},
        {"icon": "🎯", "label": "Selected Subset", "value": f"{len(selected_feats) if selected_feats else t['cols']}", "sub": "Features for modeling"},
        {"icon": "🧬", "label": "Observations", "value": f"{t['rows']:,}", "sub": "Active row count"},
        {"icon": "⚡", "label": "Engineering Status", "value": "Applied" if st.session_state.get("feature_eng_done") else "Ready", "sub": "Pipeline stage 08", "delta_type": "pos"},
    ]
    render_metric_grid(metrics)

    # Tabs for Engineering
    tab_math, tab_poly, tab_pca, tab_rfe_smote = st.tabs([
        "📈 Math Transformations",
        "✖️ Polynomial Features",
        "🌌 Principal Component Analysis (PCA)",
        "🎯 Feature Selection & SMOTE",
    ])

    # 1. Math Transforms
    with tab_math:
        render_section_header("01", "Nonlinear Transformations")
        render_html(
            """
            <div style="color: #94A3B8; font-size: 14px; margin-bottom: 14px;">
                Condense skewness and stabilize variance for continuous numerical distributions:
            </div>
            """
        )

        c_m1, c_m2 = st.columns(2)
        with c_m1:
            trans_method = st.selectbox(
                "Select Transformation Algorithm:",
                ["Log1p (Safe Logarithm with Auto-Shift)", "Box-Cox (Strict Power Transform)", "Yeo-Johnson (Supports Positive and Negative)"],
                key="math_trans_method"
            )
        with c_m2:
            target_math_cols = st.multiselect("Select Target Features:", num_cols, default=num_cols[:3] if num_cols else [], key="math_trans_cols")

        if st.button("⚡ APPLY MATHEMATICAL TRANSFORMATION", use_container_width=True, key="btn_apply_math_trans"):
            if not target_math_cols:
                st.error("Please select at least one feature.")
            else:
                m_code = "log1p"
                if "Box-Cox" in trans_method:
                    m_code = "box-cox"
                elif "Yeo-Johnson" in trans_method:
                    m_code = "yeo-johnson"

                df_trans = transform_features(df, target_math_cols, method=m_code)
                st.session_state.feature_eng_done = True
                set_active_data(df_trans, stage_name="Feature Engineering", log_entry=f"Applied {trans_method} to: {', '.join(target_math_cols)}")
                st.success(f"✓ Applied {trans_method} to {len(target_math_cols)} features!")
                st.rerun()

    # 2. Polynomial Features
    with tab_poly:
        render_section_header("02", "Polynomial Expansion & Interaction Terms")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            poly_cols = st.multiselect("Select Features for Interaction:", num_cols, default=num_cols[:2] if len(num_cols) >= 2 else num_cols, key="poly_features_select")
            poly_degree = st.slider("Polynomial Degree:", 2, 3, 2)
        with c_p2:
            interaction_only = st.checkbox("Interaction terms only (exclude single-feature powers)", value=False)
            include_bias = st.checkbox("Include bias / intercept column", value=False)

        if st.button("⚡ GENERATE POLYNOMIAL EXPANSION", use_container_width=True, key="btn_apply_poly"):
            if not poly_cols:
                st.error("Please select at least one feature.")
            else:
                try:
                    df_poly = generate_polynomial_features(df, poly_cols, degree=poly_degree, interaction_only=interaction_only, include_bias=include_bias)
                    st.session_state.feature_eng_done = True
                    set_active_data(df_poly, stage_name="Feature Engineering", log_entry=f"Generated degree-{poly_degree} polynomial features for: {', '.join(poly_cols)}")
                    st.success(f"✓ Expanded feature space to {df_poly.shape[1]} columns!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Expansion error: {e}")

    # 3. PCA
    with tab_pca:
        render_section_header("03", "Principal Component Analysis (PCA)")
        c_pca1, c_pca2 = st.columns(2)
        with c_pca1:
            pca_target_cols = st.multiselect("Select Numerical Features for PCA:", num_cols, default=num_cols, key="pca_features_select")
            standardize_pca = st.checkbox("Standardize features before PCA (Recommended)", value=True)
        with c_pca2:
            pca_mode = st.radio("PCA Reduction Mode:", ["Explained Variance Ratio Threshold (e.g. 95%)", "Fixed Number of Components (e.g. 2 or 3)"])
            if "Variance" in pca_mode:
                var_threshold = st.slider("Target Cumulative Explained Variance:", 0.50, 0.99, 0.95, step=0.01)
                n_comp = var_threshold
            else:
                n_comp = st.slider("Number of Principal Components (k):", 2, max(2, len(pca_target_cols)), 3)

        if st.button("⚡ EXECUTE PCA DIMENSIONALITY REDUCTION", use_container_width=True, key="btn_apply_pca"):
            if len(pca_target_cols) < 2:
                st.error("PCA requires at least 2 numerical features.")
            else:
                try:
                    df_pca, pca_model, var_ratios = apply_pca(df, pca_target_cols, n_components=n_comp, standardize=standardize_pca)
                    st.session_state.feature_eng_done = True
                    set_active_data(df_pca, stage_name="PCA", log_entry=f"Applied PCA: reduced {len(pca_target_cols)} features to {pca_model.n_components_} components (Cumulative Var: {var_ratios.sum()*100:.1f}%).")
                    st.success(f"✓ PCA complete! Created {pca_model.n_components_} Principal Components explaining {var_ratios.sum()*100:.1f}% of variance.")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ PCA error: {e}")

    # 4. RFE & SMOTE
    with tab_rfe_smote:
        render_section_header("04", "RFE Selection & Class Balancing")
        c_rf1, c_rf2 = st.columns(2)

        with c_rf1:
            st.markdown("#### 🎯 Recursive Feature Elimination (RFE)")
            target_candidates = df.columns.tolist()
            selected_target = st.selectbox("Select Target Variable (Y):", target_candidates, index=len(target_candidates)-1, key="rfe_target_select")
            available_features = [c for c in num_cols if c != selected_target]
            n_rfe_feats = st.slider("Target Number of Features to Keep:", 1, max(1, len(available_features)), min(5, len(available_features)))

            if st.button("🎯 RUN RFE FEATURE SELECTION", use_container_width=True, key="btn_run_rfe"):
                if not available_features:
                    st.error("No valid features for RFE.")
                else:
                    try:
                        X_in = df[available_features]
                        y_in = df[selected_target]
                        prob_type = "Regression" if pd.api.types.is_numeric_dtype(y_in) and y_in.nunique() > 10 else "Classification"
                        X_sel, sel_cols, _ = apply_rfe_selection(X_in, y_in, n_features=n_rfe_feats, problem_type=prob_type)
                        st.session_state.selected_features = sel_cols
                        st.session_state.target_column = selected_target
                        st.success(f"✓ RFE selected {len(sel_cols)} features: {', '.join(sel_cols)}")
                    except Exception as e:
                        st.error(f"❌ RFE Error: {e}")

        with c_rf2:
            st.markdown("#### ⚖️ Handle Class Imbalance (SMOTE)")
            class_target = st.selectbox("Select Classification Target:", df.columns.tolist(), index=len(df.columns)-1, key="smote_target_select")
            y_target = df[class_target]
            counts = y_target.value_counts()
            st.write("Current Class Distribution:")
            st.dataframe(counts.rename("Count"), use_container_width=True)

            smote_method = st.radio("Sampling Method:", ["SMOTE (Synthetic Minority Oversampling)", "Random Under-Sampling"])
            if st.button("⚡ BALANCE DATASET", use_container_width=True, key="btn_apply_smote"):
                try:
                    num_feats = [c for c in num_cols if c != class_target]
                    if not num_feats:
                        st.error("Balancing requires numerical features.")
                    else:
                        X_imb = df[num_feats]
                        m_code = "smote" if "SMOTE" in smote_method else "undersampling"
                        X_res, y_res = handle_imbalanced_classes(X_imb, y_target, method=m_code)
                        df_balanced = pd.concat([X_res, y_res], axis=1)
                        st.session_state.imbalance_done = True
                        set_active_data(df_balanced, stage_name="Balancing", log_entry=f"Balanced class distribution using {smote_method}. Final shape: {df_balanced.shape}.")
                        st.success(f"✓ Balanced dataset! New shape: {df_balanced.shape[0]:,} rows.")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Balancing error: {e}")

    # Before / After Matrix
    orig_df = st.session_state.get("original_data")
    if orig_df is not None:
        render_html("<br>")
        render_section_header("05", "Feature Engineering Audit")
        render_before_after(orig_df, df, note="Monitoring dimensional changes and feature expansion.")

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/07_Encoding.py",
        next_page="pages/09_Scaling.py",
        prev_label="← Categorical Encoding",
        next_label="Feature Normalization & Scaling →"
    )


if __name__ == "__main__":
    main()
