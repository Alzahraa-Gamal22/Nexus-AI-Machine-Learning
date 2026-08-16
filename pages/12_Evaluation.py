"""
Nexus AI — Page 12: Model Evaluation & Diagnostic Studio
--------------------------------------------------------
Deep performance telemetry, confusion matrices, ROC/PR curves, feature importance,
residual diagnostics, and predictions inspection.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from utils.session_state import init_session_state, get_active_data
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_step_navigation, render_sidebar_status, render_html
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve_chart,
    plot_precision_recall_chart,
    plot_feature_importance_chart,
    plot_actual_vs_predicted,
    plot_residuals_chart,
    plot_3d_pca_space,
)


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    # Guard: Model Trained?
    model = st.session_state.get("model")
    if model is None:
        render_hero(
            badge="⚠️ EVALUATION GUARD",
            title="Model Diagnostics & Performance",
            subtitle="Deep model telemetry requires a trained model in memory."
        )
        st.warning("🤖 No active trained model found in session state.")
        if st.button("🧠 Jump to Model Training Studio", use_container_width=True):
            st.switch_page("pages/11_Modeling.py")
        return

    problem_type = st.session_state.get("problem_type", "Classification")
    model_name = st.session_state.get("model_name", "Model")
    metrics = st.session_state.get("evaluation_metrics", {})
    t_dur = st.session_state.get("training_time", 0.0)

    render_hero(
        badge="● DIAGNOSTIC PROTOCOL",
        title=f"Model Evaluation: {model_name}",
        subtitle=f"Comprehensive performance audit and diagnostic benchmarks for {problem_type} architecture."
    )

    # Top KPI Metrics Grid
    m_cards = []
    for k, v in metrics.items():
        if isinstance(v, float):
            val_str = f"{v:.4f}" if "Score" in k or "R²" in k or "Silhouette" in k else f"{v:.2%}" if "Accuracy" in k or "Precision" in k or "Recall" in k or "F1" in k else f"{v:.3f}"
        else:
            val_str = str(v)
        m_cards.append({"icon": "⭐", "label": k, "value": val_str})
    render_metric_grid(m_cards[:4])

    # 1. Classification Diagnostics
    if problem_type == "Classification":
        y_test = st.session_state.get("y_test")
        y_pred = st.session_state.get("y_pred")
        y_prob = st.session_state.get("y_prob")
        imp_df = st.session_state.get("feature_importance_df")
        pred_df = st.session_state.get("predictions_df")

        tab_cm, tab_curves, tab_imp, tab_preds, tab_report = st.tabs([
            "🔮 Confusion Matrix",
            "📈 ROC & PR Curves",
            "📊 Feature Importance",
            "📋 Prediction Telemetry",
            "📄 Classification Report",
        ])

        with tab_cm:
            render_section_header("01", "Confusion Matrix Analysis")
            labels = np.unique(y_test)
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            st.plotly_chart(plot_confusion_matrix(cm, labels), use_container_width=True)

        with tab_curves:
            render_section_header("02", "Discriminative Power & Precision-Recall")
            if y_prob is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(plot_roc_curve_chart(y_test, y_prob), use_container_width=True)
                with c2:
                    st.plotly_chart(plot_precision_recall_chart(y_test, y_prob), use_container_width=True)
            else:
                st.info("ℹ️ Probability estimates not available for this model architecture (ROC/PR requires predict_proba).")

        with tab_imp:
            render_section_header("03", "Feature Importance & Model Attribution")
            if imp_df is not None and not imp_df.empty:
                st.plotly_chart(plot_feature_importance_chart(imp_df), use_container_width=True)
                st.dataframe(imp_df.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Feature importance or coefficients not available for this algorithm.")

        with tab_preds:
            render_section_header("04", "Holdout Prediction Observations")
            if pred_df is not None:
                filter_choice = st.radio("Filter Predictions:", ["All Predictions", "Correct Only", "Misclassified Only"], horizontal=True)
                if filter_choice == "Correct Only":
                    st.dataframe(pred_df[pred_df["Accurate"] == True], use_container_width=True)
                elif filter_choice == "Misclassified Only":
                    st.dataframe(pred_df[pred_df["Accurate"] == False], use_container_width=True)
                else:
                    st.dataframe(pred_df, use_container_width=True)

        with tab_report:
            render_section_header("05", "Detailed Precision, Recall & F1 Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report_dict).T.round(4), use_container_width=True)

    # 2. Regression Diagnostics
    elif problem_type == "Regression":
        y_test = st.session_state.get("y_test")
        y_pred = st.session_state.get("y_pred")
        imp_df = st.session_state.get("feature_importance_df")
        pred_df = st.session_state.get("predictions_df")

        tab_scat, tab_res, tab_imp, tab_preds = st.tabs([
            "🎯 Actual vs Predicted",
            "📉 Residuals Diagnostics",
            "📊 Feature Importance",
            "📋 Prediction Telemetry",
        ])

        with tab_scat:
            render_section_header("01", "Actual Ground Truth vs Model Prediction")
            st.plotly_chart(plot_actual_vs_predicted(y_test, y_pred), use_container_width=True)

        with tab_res:
            render_section_header("02", "Residual Error Diagnostics")
            st.plotly_chart(plot_residuals_chart(y_test, y_pred), use_container_width=True)

        with tab_imp:
            render_section_header("03", "Feature Impact & Magnitude")
            if imp_df is not None and not imp_df.empty:
                st.plotly_chart(plot_feature_importance_chart(imp_df), use_container_width=True)
                st.dataframe(imp_df.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Feature weights not available for this regressor.")

        with tab_preds:
            render_section_header("04", "Holdout Predictions & Error Table")
            if pred_df is not None:
                st.dataframe(pred_df, use_container_width=True)

    # 3. Clustering Diagnostics
    elif problem_type == "Clustering":
        labels = st.session_state.get("cluster_labels")
        X = get_active_data()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        tab_dist, tab_pca3d, tab_data = st.tabs([
            "🧩 Cluster Distribution",
            "🌌 3D PCA Cluster Manifold",
            "📋 Clustered Dataset",
        ])

        with tab_dist:
            render_section_header("01", "Observation Density per Cluster")
            c_counts = pd.Series(labels).value_counts().sort_index().reset_index()
            c_counts.columns = ["Cluster ID", "Samples Count"]
            st.dataframe(c_counts, use_container_width=True, hide_index=True)

        with tab_pca3d:
            render_section_header("02", "3D PCA Cluster Space Projection")
            if len(num_cols) >= 3:
                pca = PCA(n_components=3)
                comps = pca.fit_transform(X[num_cols].dropna())
                pca_df = pd.DataFrame({
                    "PC_1": comps[:, 0],
                    "PC_2": comps[:, 1],
                    "PC_3": comps[:, 2],
                    "Cluster": labels.astype(str)
                })
                fig_3d_cluster = plot_3d_pca_space(pca_df, color_col="Cluster", explained_variance=pca.explained_variance_ratio_)
                st.plotly_chart(fig_3d_cluster, use_container_width=True)
            else:
                st.info("At least 3 features required for 3D PCA cluster visualization.")

        with tab_data:
            render_section_header("03", "Dataset with Assigned Cluster Labels")
            clustered_df = X.copy()
            clustered_df["Cluster_Label"] = labels
            st.dataframe(clustered_df.head(50), use_container_width=True)

    # Model Parameters Card
    render_html("<br>")
    with st.expander("📦 Inspect Model Internal Hyperparameters & Architecture"):
        st.code(str(model), language="python")
        if hasattr(model, "get_params"):
            st.json(model.get_params())

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/11_Modeling.py",
        next_page="pages/13_Export.py",
        prev_label="← Model Studio",
        next_label="AI Summary & Export Center →"
    )


if __name__ == "__main__":
    main()
