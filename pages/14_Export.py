"""
Nexus AI — Page 13: Export Center & AI Analysis Audit Report
------------------------------------------------------------
Multi-format downloads for processed datasets (CSV, Excel), predictions,
model metrics (JSON), full AI Audit Reports, and serialized model binaries.
"""

import streamlit as st
import pandas as pd
from utils.session_state import init_session_state, get_active_data, get_dataset_telemetry
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_step_navigation, render_sidebar_status, dataset_guard, render_html
from utils.export import to_csv_bytes, to_excel_bytes, to_json_bytes, to_joblib_bytes, generate_audit_report


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    if not dataset_guard():
        return

    df = get_active_data()
    t = get_dataset_telemetry(df)
    dataset_name = st.session_state.get("dataset_name", "nexus_data")
    model = st.session_state.get("model")
    model_name = st.session_state.get("model_name", "model")
    pred_df = st.session_state.get("predictions_df")
    metrics = st.session_state.get("evaluation_metrics", {})
    imp_df = st.session_state.get("feature_importance_df")

    render_hero(
        badge="● EXPORT CENTER & AUDIT",
        title="AI Analysis Summary & Export Hub",
        subtitle="Export transformed datasets, model predictions, evaluation benchmarks, full audit reports, and serialized model binaries."
    )

    # Top KPI Metrics Grid
    metrics_cards = [
        {"icon": "🧬", "label": "Processed Rows", "value": f"{t['rows']:,}", "sub": f"{t['cols']} total features"},
        {"icon": "🩹", "label": "Missing Values", "value": f"{t['missing_count']:,}", "sub": f"{t['missing_pct']:.1f}% null rate", "delta_type": "pos"},
        {"icon": "🤖", "label": "Trained Architecture", "value": str(model_name)[:16], "sub": st.session_state.get("problem_type", "ML")},
        {"icon": "💾", "label": "Artifact Hub", "value": "Ready", "sub": "Multi-format download", "delta_type": "pos"},
    ]
    render_metric_grid(metrics_cards)

    # Export Panels Grid
    col_exp1, col_exp2 = st.columns(2)

    # 1. Dataset Downloads
    with col_exp1:
        with st.container(border=True):
            render_section_header("01", "Processed Dataset Exports")
            render_html(
                f"""
                <div style="color: #94A3B8; font-size: 13px; margin-bottom: 14px;">
                    Download the final preprocessed, cleaned, encoded, and scaled dataset ({t['rows']:,} rows × {t['cols']} columns):
                </div>
                """
            )

            c_d1, c_d2 = st.columns(2)
            with c_d1:
                csv_data = to_csv_bytes(df)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"{dataset_name}_processed.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_dl_csv"
                )
            with c_d2:
                try:
                    excel_data = to_excel_bytes(df)
                    st.download_button(
                        label="📥 Download Excel (.xlsx)",
                        data=excel_data,
                        file_name=f"{dataset_name}_processed.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="btn_dl_xlsx"
                    )
                except Exception as e:
                    st.caption(f"Excel export note: {e}")

    # 2. Predictions Downloads
    with col_exp2:
        with st.container(border=True):
            render_section_header("02", "Predictions & Test Results")
            if pred_df is not None:
                render_html(
                    f"""
                    <div style="color: #94A3B8; font-size: 13px; margin-bottom: 14px;">
                        Download model holdout test predictions and classification/regression telemetry ({len(pred_df):,} samples):
                    </div>
                    """
                )
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    pred_csv = to_csv_bytes(pred_df)
                    st.download_button(
                        label="📥 Download Predictions (CSV)",
                        data=pred_csv,
                        file_name=f"{dataset_name}_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="btn_dl_pred_csv"
                    )
                with c_p2:
                    try:
                        pred_xlsx = to_excel_bytes(pred_df)
                        st.download_button(
                            label="📥 Download Predictions (Excel)",
                            data=pred_xlsx,
                            file_name=f"{dataset_name}_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="btn_dl_pred_xlsx"
                        )
                    except Exception as e:
                        st.caption(f"Excel export note: {e}")
            else:
                st.info("ℹ️ Train a model in the Modeling Studio to generate downloadable predictions.")

    # 3. Model Binary & Metrics
    render_html("<br>")
    col_mod1, col_mod2 = st.columns(2)

    with col_mod1:
        with st.container(border=True):
            render_section_header("03", "Serialized Model & Weights")
            if model is not None:
                render_html(
                    """
                    <div style="color: #94A3B8; font-size: 13px; margin-bottom: 14px;">
                        Export the trained Scikit-Learn / XGBoost model artifact for Python production deployment:
                    </div>
                    """
                )
                try:
                    joblib_bytes = to_joblib_bytes(model)
                    st.download_button(
                        label=f"📦 Download Model ({model_name}.joblib)",
                        data=joblib_bytes,
                        file_name=f"{dataset_name}_{model_name.lower().replace(' ', '_')}.joblib",
                        mime="application/octet-stream",
                        use_container_width=True,
                        key="btn_dl_joblib"
                    )
                except Exception as e:
                    st.error(f"Failed to serialize model: {e}")
            else:
                st.info("ℹ️ Train a model in the Modeling Studio to enable model binary download.")

    with col_mod2:
        with st.container(border=True):
            render_section_header("04", "Evaluation Metrics & Parameters (JSON)")
            if metrics:
                render_html(
                    """
                    <div style="color: #94A3B8; font-size: 13px; margin-bottom: 14px;">
                        Download comprehensive performance metrics and model hyperparameters in structured JSON format:
                    </div>
                    """
                )
                json_payload = {
                    "dataset": dataset_name,
                    "model_name": model_name,
                    "problem_type": st.session_state.get("problem_type"),
                    "target_column": st.session_state.get("target_column"),
                    "evaluation_metrics": metrics,
                    "transformation_log": st.session_state.get("transformation_log", []),
                }
                if hasattr(model, "get_params"):
                    try:
                        json_payload["hyperparameters"] = model.get_params()
                    except Exception:
                        pass

                json_bytes = to_json_bytes(json_payload)
                st.download_button(
                    label="📄 Download Metrics JSON",
                    data=json_bytes,
                    file_name=f"{dataset_name}_metrics.json",
                    mime="application/json",
                    use_container_width=True,
                    key="btn_dl_json"
                )
            else:
                st.info("ℹ️ Evaluation metrics will be available once model evaluation is complete.")

    # 4. Comprehensive AI Audit Report
    render_html("<br>")
    with st.container(border=True):
        render_section_header("05", "Comprehensive AI Analysis Audit Report")
        report_text = generate_audit_report()

        c_rep1, c_rep2 = st.columns([1, 1])
        with c_rep1:
            st.download_button(
                label="📄 Download Audit Report (Markdown)",
                data=report_text.encode("utf-8"),
                file_name=f"{dataset_name}_AI_Audit_Report.md",
                mime="text/markdown",
                use_container_width=True,
                key="btn_dl_rep_md"
            )
        with c_rep2:
            st.download_button(
                label="📄 Download Audit Report (Plain Text)",
                data=report_text.encode("utf-8"),
                file_name=f"{dataset_name}_AI_Audit_Report.txt",
                mime="text/plain",
                use_container_width=True,
                key="btn_dl_rep_txt"
            )

        with st.expander("👁️ Live Preview: AI Audit Report", expanded=False):
            st.markdown(report_text)

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/12_Evaluation.py",
        next_page="pages/01_Dashboard.py",
        prev_label="← Model Evaluation & Testing",
        next_label="Return to Dashboard 🚀"
    )


if __name__ == "__main__":
    main()
