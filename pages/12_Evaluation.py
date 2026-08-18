"""
Nexus AI — Page 12: Model Evaluation, Diagnostics & Live Testing Studio
------------------------------------------------------------------------
Unified studio for deep performance telemetry, confusion matrices, ROC/PR curves,
residual diagnostics, feature attribution, single-sample RAW prediction, and
unseen batch raw dataset testing.
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.decomposition import PCA

from utils.session_state import init_session_state, get_active_data
from utils.preprocessing import PreprocessingPipeline
from utils.ui import (
    inject_global_css,
    render_hero,
    render_section_header,
    render_metric_grid,
    render_step_navigation,
    render_sidebar_status,
    render_html,
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve_chart,
    plot_precision_recall_chart,
    plot_feature_importance_chart,
    plot_actual_vs_predicted,
    plot_residuals_chart,
    plot_3d_pca_space,
    plot_prediction_probabilities,
)


def render_live_testing_hub(pipeline: PreprocessingPipeline):
    """
    Consolidated Live Testing Hub:
    1. Single Sample Raw Input Form (Natural Values)
    2. Batch Raw Test Dataset Upload (CSV/Excel) with auto-evaluation
    3. Ground Truth vs Prediction Inspector
    """
    model = st.session_state.get("model")
    problem_type = st.session_state.get("problem_type", "Classification")
    target_col = st.session_state.get("target_column")
    df = get_active_data()
    orig_df = st.session_state.get("original_data")
    ref_raw_df = orig_df if orig_df is not None else df

    required_raw_features = pipeline.get_required_raw_features() if pipeline else []
    if not required_raw_features and ref_raw_df is not None:
        required_raw_features = [c for c in ref_raw_df.columns if c != target_col]

    if not required_raw_features:
        st.warning("⚠️ No input features identified for live testing.")
        return

    sub_single, sub_batch, sub_inspect = st.tabs([
        "🎛️ Single Sample Live Tester (Raw Inputs)",
        "📁 Batch Raw Test Dataset Upload (CSV/Excel)",
        "🔍 Inspect Dataset Observations",
    ])

    # =========================================================================
    # TAB 1: SINGLE SAMPLE RAW INPUT TESTER
    # =========================================================================
    with sub_single:
        with st.container(border=True):
            render_section_header("01", "Interactive Raw Feature Input Form")
            st.caption("Enter natural, unencoded, unscaled values for each original feature. Nexus AI will automatically apply the trained preprocessing pipeline.")

            # Quick Preset Actions
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                if st.button("🎲 Random Sample from Dataset", use_container_width=True, key="btn_single_rand"):
                    if ref_raw_df is not None and len(ref_raw_df) > 0:
                        rand_row = ref_raw_df.sample(n=1, random_state=np.random.randint(0, 100000)).iloc[0]
                        for feat in required_raw_features:
                            if feat in rand_row:
                                st.session_state[f"raw_in_{feat}"] = rand_row[feat]
                        st.rerun()

            with col_b2:
                if st.button("📊 Set to Feature Means / Defaults", use_container_width=True, key="btn_single_means"):
                    for feat in required_raw_features:
                        if feat in pipeline.raw_numeric_stats:
                            st.session_state[f"raw_in_{feat}"] = float(pipeline.raw_numeric_stats[feat]["mean"])
                        elif feat in pipeline.raw_categorical_values and pipeline.raw_categorical_values[feat]:
                            st.session_state[f"raw_in_{feat}"] = pipeline.raw_categorical_values[feat][0]
                    st.rerun()

            with col_b3:
                if st.button("🔄 Set to Feature Minimums", use_container_width=True, key="btn_single_mins"):
                    for feat in required_raw_features:
                        if feat in pipeline.raw_numeric_stats:
                            st.session_state[f"raw_in_{feat}"] = float(pipeline.raw_numeric_stats[feat]["min"])
                        elif feat in pipeline.raw_categorical_values and pipeline.raw_categorical_values[feat]:
                            st.session_state[f"raw_in_{feat}"] = pipeline.raw_categorical_values[feat][0]
                    st.rerun()

            # Dynamic Input Grid
            num_cols_grid = 3 if len(required_raw_features) >= 3 else max(1, len(required_raw_features))
            grid_cols = st.columns(num_cols_grid)
            single_raw_input = {}

            for idx, feat in enumerate(required_raw_features):
                col_target = grid_cols[idx % num_cols_grid]
                input_key = f"raw_in_{feat}"

                with col_target:
                    # Check if categorical or numeric
                    is_cat = False
                    cat_options = []
                    if feat in pipeline.raw_categorical_values and pipeline.raw_categorical_values[feat]:
                        is_cat = True
                        cat_options = pipeline.raw_categorical_values[feat]
                    elif ref_raw_df is not None and feat in ref_raw_df.columns and not pd.api.types.is_numeric_dtype(ref_raw_df[feat]):
                        is_cat = True
                        cat_options = [str(x) for x in ref_raw_df[feat].dropna().unique()]

                    if is_cat:
                        if not cat_options:
                            cat_options = ["Standard", "Other"]
                        prev_val = str(st.session_state.get(input_key, cat_options[0]))
                        def_idx = cat_options.index(prev_val) if prev_val in cat_options else 0
                        sel_cat = st.selectbox(
                            label=f"**{feat}** *(Categorical)*",
                            options=cat_options,
                            index=def_idx,
                            key=input_key,
                            help=f"Options: {', '.join(cat_options[:5])}"
                        )
                        single_raw_input[feat] = sel_cat

                    else:
                        # Numeric feature
                        stats = pipeline.raw_numeric_stats.get(feat, {})
                        f_min = stats.get("min", 0.0)
                        f_max = stats.get("max", 1000.0)
                        f_mean = stats.get("mean", 0.0)
                        f_median = stats.get("median", f_mean)
                        step_val = max((f_max - f_min) / 100.0, 0.01) if f_max > f_min else 0.1

                        if input_key not in st.session_state:
                            st.session_state[input_key] = float(f_median)

                        curr_val = float(st.session_state[input_key])
                        val = st.number_input(
                            label=f"**{feat}** *(Numeric)*",
                            value=curr_val,
                            step=float(round(step_val, 4)),
                            format="%.4f" if step_val < 0.1 else "%.2f",
                            key=input_key,
                            help=f"Training Range: [{f_min:.2f} → {f_max:.2f}] | Median: {f_median:.2f}"
                        )
                        single_raw_input[feat] = val

            render_html("<br>")
            if st.button("🔮 RUN INTELLIGENT MODEL INFERENCE", type="primary", use_container_width=True, key="btn_run_single_raw_pred"):
                raw_input_df = pd.DataFrame([single_raw_input])

                try:
                    # Validate raw input
                    is_valid, errors, warnings = pipeline.validate_raw_input(raw_input_df)
                    if not is_valid:
                        for err in errors:
                            st.error(f"❌ Validation Error: {err}")
                    else:
                        if warnings:
                            for w in warnings:
                                st.caption(f"⚠️ {w}")

                        # Transform through full training preprocessing pipeline
                        transformed_input = pipeline.transform(raw_input_df)

                        # Execute Model Prediction
                        raw_pred = model.predict(transformed_input)
                        decoded_pred = pipeline.inverse_transform_target(raw_pred)[0]

                        # Render Inference Result Card
                        render_html("<br>")
                        with st.container(border=True):
                            render_section_header("02", "Instant Inference Telemetry")

                            if problem_type == "Classification":
                                col_r1, col_r2 = st.columns([1, 1.5])
                                with col_r1:
                                    render_html(f"""
                                    <div style="background: linear-gradient(135deg, rgba(56,189,248,0.15), rgba(139,92,246,0.15)); border: 1px solid rgba(56,189,248,0.4); border-radius: 14px; padding: 22px; text-align: center;">
                                        <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 6px;">🎯 PREDICTED CLASS</div>
                                        <div style="color: #38BDF8; font-size: 2.2rem; font-weight: 900; text-shadow: 0 0 16px rgba(56,189,248,0.4);">{decoded_pred}</div>
                                    </div>
                                    """)
                                    if hasattr(model, "predict_proba"):
                                        probs = model.predict_proba(transformed_input)[0]
                                        max_prob = np.max(probs)
                                        render_html(f"""
                                        <div style="margin-top: 10px; background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 12px; text-align: center;">
                                            <span style="color: #CBD5E1; font-size: 0.9rem;">Confidence Score: </span>
                                            <span style="color: #10B981; font-size: 1.2rem; font-weight: 800;">{max_prob:.2%}</span>
                                        </div>
                                        """)

                                with col_r2:
                                    if hasattr(model, "predict_proba"):
                                        probs = model.predict_proba(transformed_input)[0]
                                        classes = pipeline.target_classes_ if pipeline.target_classes_ is not None else getattr(model, "classes_", [f"Class {i}" for i in range(len(probs))])
                                        st.plotly_chart(plot_prediction_probabilities(classes, probs), use_container_width=True)
                                    else:
                                        st.info("ℹ️ Prediction generated successfully. (Model does not support predict_proba curve).")

                            elif problem_type == "Regression":
                                col_r1, col_r2 = st.columns(2)
                                with col_r1:
                                    render_html(f"""
                                    <div style="background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(56,189,248,0.15)); border: 1px solid rgba(16,185,129,0.4); border-radius: 14px; padding: 22px; text-align: center;">
                                        <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 6px;">🎯 PREDICTED VALUE ({target_col or 'Target'})</div>
                                        <div style="color: #10B981; font-size: 2.4rem; font-weight: 900; text-shadow: 0 0 16px rgba(16,185,129,0.4);">{float(decoded_pred):.4f}</div>
                                    </div>
                                    """)
                                with col_r2:
                                    if target_col and target_col in pipeline.raw_numeric_stats:
                                        t_stat = pipeline.raw_numeric_stats[target_col]
                                        render_html(f"""
                                        <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px;">
                                            <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 700; margin-bottom: 10px;">📊 TRAINING TARGET DISTRIBUTION</div>
                                            <div style="display: flex; justify-content: space-between; margin-bottom: 6px; color: #CBD5E1;">
                                                <span>Min: <b>{t_stat['min']:.2f}</b></span>
                                                <span>Median: <b>{t_stat['median']:.2f}</b></span>
                                                <span>Max: <b>{t_stat['max']:.2f}</b></span>
                                            </div>
                                        </div>
                                        """)

                            elif problem_type == "Clustering":
                                render_html(f"""
                                <div style="background: linear-gradient(135deg, rgba(139,92,246,0.15), rgba(236,72,153,0.15)); border: 1px solid rgba(139,92,246,0.4); border-radius: 14px; padding: 22px; text-align: center;">
                                    <div style="color: #94A3B8; font-size: 0.85rem; font-weight: 700; letter-spacing: 1px; margin-bottom: 6px;">🧩 ASSIGNED CLUSTER</div>
                                    <div style="color: #A855F7; font-size: 2.4rem; font-weight: 900; text-shadow: 0 0 16px rgba(168,85,247,0.4);">Cluster {decoded_pred}</div>
                                </div>
                                """)

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

    # =========================================================================
    # TAB 2: BATCH RAW TEST DATASET UPLOAD
    # =========================================================================
    with sub_batch:
        with st.container(border=True):
            render_section_header("01", "Upload Unseen Raw Test Dataset (CSV / Excel)")
            st.caption("Upload a new file containing raw unseen test data. Nexus AI will validate features, apply the full training preprocessing pipeline automatically, evaluate test metrics (if target column is present), and generate downloadable predictions.")

            uploaded_test_file = st.file_uploader(
                "Upload Raw Test File:",
                type=["csv", "xlsx", "xls"],
                key="batch_raw_test_file_uploader"
            )

            if uploaded_test_file is not None:
                try:
                    if uploaded_test_file.name.lower().endswith(".csv"):
                        batch_raw_df = pd.read_csv(uploaded_test_file)
                    else:
                        batch_raw_df = pd.read_excel(uploaded_test_file)

                    st.info(f"📄 Loaded file: `{uploaded_test_file.name}` ({batch_raw_df.shape[0]:,} rows, {batch_raw_df.shape[1]} columns)")

                    # Validate raw features
                    is_valid, errors, warnings = pipeline.validate_raw_input(batch_raw_df, require_target=False)

                    if not is_valid:
                        for err in errors:
                            st.error(f"❌ Schema Error: {err}")
                    else:
                        if warnings:
                            for w in warnings:
                                st.warning(f"⚠️ {w}")

                        has_target = bool(target_col and target_col in batch_raw_df.columns)

                        if has_target:
                            st.success(f"🎯 Target column `{target_col}` detected in test file! System will calculate unseen holdout evaluation benchmarks.")
                        else:
                            st.info("ℹ️ No target column detected. System will operate in prediction-only inference mode.")

                        render_html("<br>")
                        if st.button("🚀 EXECUTE BATCH TEST PIPELINE & PREDICTIONS", type="primary", use_container_width=True, key="btn_run_batch_test"):
                            with st.spinner("⚡ Applying preprocessing pipeline and generating predictions..."):
                                # 1. Preprocess raw data through pipeline
                                batch_transformed = pipeline.transform(batch_raw_df)

                                # 2. Model Prediction
                                batch_raw_preds = model.predict(batch_transformed)
                                batch_decoded_preds = pipeline.inverse_transform_target(batch_raw_preds)

                                # 3. Construct Result DataFrame
                                out_df = batch_raw_df.copy()
                                pred_col_name = f"Predicted_{target_col or 'Target'}"
                                out_df[pred_col_name] = batch_decoded_preds

                                if hasattr(model, "predict_proba") and problem_type == "Classification":
                                    probs = model.predict_proba(batch_transformed)
                                    out_df["Confidence"] = [f"{np.max(p):.2%}" for p in probs]

                                # 4. If Target Exists: Compute Unseen Test Metrics
                                if has_target and problem_type == "Classification":
                                    y_true_raw = batch_raw_df[target_col]
                                    test_acc = accuracy_score(y_true_raw.astype(str), pd.Series(batch_decoded_preds).astype(str))
                                    test_f1 = f1_score(y_true_raw.astype(str), pd.Series(batch_decoded_preds).astype(str), average="weighted", zero_division=0)
                                    test_prec = precision_score(y_true_raw.astype(str), pd.Series(batch_decoded_preds).astype(str), average="weighted", zero_division=0)
                                    test_rec = recall_score(y_true_raw.astype(str), pd.Series(batch_decoded_preds).astype(str), average="weighted", zero_division=0)

                                    out_df["Accurate"] = y_true_raw.astype(str) == pd.Series(batch_decoded_preds).astype(str)

                                    render_html("<br>")
                                    render_section_header("02", "Unseen Test Benchmark Telemetry")
                                    test_cards = [
                                        {"icon": "🎯", "label": "Test Accuracy", "value": f"{test_acc:.2%}", "sub": "Unseen Dataset", "delta_type": "pos"},
                                        {"icon": "⚖️", "label": "Test F1 Score", "value": f"{test_f1:.2%}", "sub": "Weighted Average", "delta_type": "pos"},
                                        {"icon": "🎯", "label": "Test Precision", "value": f"{test_prec:.2%}", "sub": "Weighted Precision"},
                                        {"icon": "🔍", "label": "Test Recall", "value": f"{test_rec:.2%}", "sub": "Weighted Recall"},
                                    ]
                                    render_metric_grid(test_cards)

                                elif has_target and problem_type == "Regression":
                                    y_true_raw = batch_raw_df[target_col].to_numpy(dtype=float)
                                    y_pred_raw = np.asarray(batch_decoded_preds, dtype=float)
                                    t_r2 = r2_score(y_true_raw, y_pred_raw)
                                    t_mae = mean_absolute_error(y_true_raw, y_pred_raw)
                                    t_rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))

                                    out_df["Absolute Error"] = np.abs(y_true_raw - y_pred_raw)

                                    render_html("<br>")
                                    render_section_header("02", "Unseen Test Benchmark Telemetry")
                                    test_cards = [
                                        {"icon": "🎯", "label": "Test R² Score", "value": f"{t_r2:.4f}", "sub": "Unseen Dataset", "delta_type": "pos"},
                                        {"icon": "📉", "label": "Test RMSE", "value": f"{t_rmse:.4f}", "sub": "Root Mean Squared Error", "delta_type": "pos"},
                                        {"icon": "📏", "label": "Test MAE", "value": f"{t_mae:.4f}", "sub": "Mean Absolute Error"},
                                        {"icon": "🧬", "label": "Test Samples", "value": f"{len(batch_raw_df):,}", "sub": "Evaluated Rows"},
                                    ]
                                    render_metric_grid(test_cards)

                                render_html("<br>")
                                render_section_header("03", "Predictions Matrix")
                                st.dataframe(out_df, use_container_width=True)

                                # Download Actions
                                render_html("<br>")
                                col_dl1, col_dl2 = st.columns(2)
                                with col_dl1:
                                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                                    st.download_button(
                                        label="📥 Download Predictions as CSV",
                                        data=csv_bytes,
                                        file_name=f"predicted_{uploaded_test_file.name}.csv",
                                        mime="text/csv",
                                        use_container_width=True,
                                        key="btn_dl_batch_csv"
                                    )
                                with col_dl2:
                                    excel_buf = io.BytesIO()
                                    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                                        out_df.to_excel(writer, index=False)
                                    st.download_button(
                                        label="📥 Download Predictions as Excel (.xlsx)",
                                        data=excel_buf.getvalue(),
                                        file_name=f"predicted_{uploaded_test_file.name}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True,
                                        key="btn_dl_batch_xlsx"
                                    )

                except Exception as e:
                    st.error(f"❌ Error processing test dataset: {e}")

    # =========================================================================
    # TAB 3: INSPECT DATASET OBSERVATIONS
    # =========================================================================
    with sub_inspect:
        with st.container(border=True):
            render_section_header("01", "Ground Truth vs Model Prediction Inspector")
            if ref_raw_df is not None and not ref_raw_df.empty:
                row_idx = st.slider("Select Observation Row Index:", 0, len(ref_raw_df) - 1, 0, key="inspect_row_slider_eval")
                row_data = ref_raw_df.iloc[row_idx]

                col_i1, col_i2 = st.columns([1.5, 1])
                with col_i1:
                    st.write(f"**Observation #{row_idx} Raw Features:**")
                    st.dataframe(pd.DataFrame([row_data[required_raw_features]]), use_container_width=True)

                with col_i2:
                    try:
                        row_raw_df = pd.DataFrame([row_data[required_raw_features]])
                        row_trans = pipeline.transform(row_raw_df)
                        row_pred = model.predict(row_trans)
                        row_dec = pipeline.inverse_transform_target(row_pred)[0]

                        if problem_type in ["Classification", "Regression"] and target_col in ref_raw_df.columns:
                            actual_val = row_data[target_col]
                            is_match = (str(row_dec) == str(actual_val)) if problem_type == "Classification" else None

                            render_html(f"""
                            <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 14px;">
                                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 700;">GROUND TRUTH (ACTUAL):</div>
                                <div style="color: #F8FAFC; font-size: 1.3rem; font-weight: 800; margin-bottom: 8px;">{actual_val}</div>
                                <div style="color: #94A3B8; font-size: 0.8rem; font-weight: 700;">MODEL PREDICTION:</div>
                                <div style="color: #38BDF8; font-size: 1.3rem; font-weight: 800; margin-bottom: 8px;">{row_dec}</div>
                                {f'<span class="metric-badge-delta pos">✓ Accurate Match</span>' if is_match else f'<span class="metric-badge-delta neg">✗ Misclassified</span>' if is_match is False else ''}
                            </div>
                            """)
                        else:
                            st.write(f"**Predicted Output:** `{row_dec}`")
                    except Exception as e:
                        st.error(f"Error inspecting row: {e}")
            else:
                st.info("No active dataset in memory to inspect.")


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
            subtitle="Deep model telemetry and testing requires a trained model in memory."
        )
        st.warning("🤖 No active trained model found in session state.")
        if st.button("🧠 Jump to Model Training Studio", use_container_width=True):
            st.switch_page("pages/11_Modeling.py")
        return

    problem_type = st.session_state.get("problem_type", "Classification")
    model_name = st.session_state.get("model_name", "Trained Model")
    metrics = st.session_state.get("evaluation_metrics", {})
    t_dur = st.session_state.get("training_time", 0.0)
    target_col = st.session_state.get("target_column")
    selected_features = st.session_state.get("selected_features", [])
    df = get_active_data()
    orig_df = st.session_state.get("original_data")

    # Get or reconstruct PreprocessingPipeline
    pipeline = st.session_state.get("preprocessing_pipeline")
    if pipeline is None:
        pipeline = PreprocessingPipeline()
        if orig_df is not None:
            pipeline.init_from_raw(orig_df, target_col=target_col, problem_type=problem_type)
        else:
            pipeline.init_from_raw(df, target_col=target_col, problem_type=problem_type)
        pipeline.record_model_features(selected_features, target_col=target_col, problem_type=problem_type)
        st.session_state.preprocessing_pipeline = pipeline

    render_hero(
        badge="● DIAGNOSTIC & TESTING PROTOCOL",
        title=f"Model Evaluation & Live Testing: {model_name}",
        subtitle=f"Comprehensive performance audit, diagnostic benchmarks, and raw-data inference lab for {problem_type} architecture."
    )

    # Top KPI Metrics Grid
    m_cards = []
    for k, v in metrics.items():
        if isinstance(v, float):
            val_str = f"{v:.4f}" if "Score" in k or "R²" in k or "Index" in k or "Silhouette" in k else f"{v:.2%}" if "Accuracy" in k or "Precision" in k or "Recall" in k or "F1" in k else f"{v:.3f}"
        else:
            val_str = str(v)
        m_cards.append({"icon": "⭐", "label": k, "value": val_str})
    render_metric_grid(m_cards[:4])

    render_html("<br>")

    # =========================================================================
    # SECTION 1: MODEL & PIPELINE ARCHITECTURE OVERVIEW
    # =========================================================================
    with st.container(border=True):
        render_section_header("01", "Model Architecture & Preprocessing Pipeline Overview")
        c_ov1, c_ov2, c_ov3, c_ov4 = st.columns(4)
        c_ov1.metric("Model Architecture", model_name, delta=problem_type)
        c_ov2.metric("Target Variable", str(target_col or "Cluster ID"))
        c_ov3.metric("Input Features", f"{len(selected_features)} Features", delta=f"{len(pipeline.raw_columns)} Raw Cols")
        train_samples = metrics.get("Train Samples", len(df) if df is not None else "--")
        test_samples = metrics.get("Test Samples", "--")
        c_ov4.metric("Dataset Split", f"{train_samples} Train / {test_samples} Test")

        # Display Pipeline Applied Stages Pills
        stages_applied = pipeline.applied_stages if pipeline.applied_stages else ["Direct Features"]
        pills_html = []
        for stg in stages_applied:
            pills_html.append(f'<span style="display:inline-block; background: rgba(56,189,248,0.12); border: 1px solid rgba(56,189,248,0.3); color: #38BDF8; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; margin-right: 8px; margin-top: 6px;">✓ {stg}</span>')

        render_html(f"""
        <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.06);">
            <span style="color: #94A3B8; font-size: 13px; font-weight: 700;">ACTIVE PREPROCESSING STEPS APPLIED TO RAW INPUTS:</span>
            <div style="margin-top: 4px;">{"".join(pills_html)}</div>
        </div>
        """)

    render_html("<br>")

    # =========================================================================
    # SECTION 2: PERFORMANCE METRICS & DEEP DIAGNOSTICS
    # =========================================================================
    if problem_type == "Classification":
        y_test = st.session_state.get("y_test")
        y_pred = st.session_state.get("y_pred")
        y_prob = st.session_state.get("y_prob")
        imp_df = st.session_state.get("feature_importance_df")
        pred_df = st.session_state.get("predictions_df")

        tab_diag_cm, tab_diag_curves, tab_diag_imp, tab_diag_preds, tab_diag_report = st.tabs([
            "🔮 Confusion Matrix",
            "📈 ROC & PR Curves",
            "📊 Feature Importance",
            "📋 Holdout Predictions",
            "📄 Classification Report",
        ])

        with tab_diag_cm:
            render_section_header("02", "Confusion Matrix Heatmap")
            if y_test is not None and y_pred is not None:
                labels = np.unique(y_test)
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                label_names = [str(pipeline.inverse_transform_target([l])[0]) if pipeline.target_encoder is not None else str(l) for l in labels]
                st.plotly_chart(plot_confusion_matrix(cm, label_names), use_container_width=True)

        with tab_diag_curves:
            render_section_header("02", "Discriminative Power & Precision-Recall")
            if y_prob is not None and y_test is not None:
                class_names = None
                if pipeline.target_classes_ is not None:
                    class_names = [str(c) for c in pipeline.target_classes_]
                elif pipeline.target_encoder is not None:
                    class_names = [str(c) for c in pipeline.target_encoder.classes_]

                c1, c2 = st.columns(2)
                with c1:
                    fig_roc = plot_roc_curve_chart(y_test, y_prob, class_names=class_names)
                    if fig_roc is not None:
                        st.plotly_chart(fig_roc, use_container_width=True)
                    else:
                        st.warning("⚠️ Receiver Operating Characteristic (ROC) Curve is unavailable for the current target/model configuration.")
                with c2:
                    fig_pr = plot_precision_recall_chart(y_test, y_prob, class_names=class_names)
                    if fig_pr is not None:
                        st.plotly_chart(fig_pr, use_container_width=True)
                    else:
                        st.warning("⚠️ Precision-Recall Curve is unavailable for the current target/model configuration.")
            else:
                st.info("ℹ️ Continuous probability or decision scores are not available for this model architecture (ROC/PR requires predict_proba or decision_function).")

        with tab_diag_imp:
            render_section_header("02", "Feature Importance & Model Attribution")
            if imp_df is not None and not imp_df.empty:
                st.plotly_chart(plot_feature_importance_chart(imp_df), use_container_width=True)
                st.dataframe(imp_df.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Feature importance or coefficients not available for this algorithm.")

        with tab_diag_preds:
            render_section_header("02", "Holdout Prediction Observations")
            if pred_df is not None:
                filter_choice = st.radio("Filter Predictions:", ["All Predictions", "Correct Only", "Misclassified Only"], horizontal=True)
                if filter_choice == "Correct Only":
                    st.dataframe(pred_df[pred_df["Accurate"] == True], use_container_width=True)
                elif filter_choice == "Misclassified Only":
                    st.dataframe(pred_df[pred_df["Accurate"] == False], use_container_width=True)
                else:
                    st.dataframe(pred_df, use_container_width=True)

        with tab_diag_report:
            render_section_header("02", "Detailed Precision, Recall & F1 Classification Report")
            if y_test is not None and y_pred is not None:
                report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                st.dataframe(pd.DataFrame(report_dict).T.round(4), use_container_width=True)

    elif problem_type == "Regression":
        y_test = st.session_state.get("y_test")
        y_pred = st.session_state.get("y_pred")
        imp_df = st.session_state.get("feature_importance_df")
        pred_df = st.session_state.get("predictions_df")

        tab_scat, tab_res, tab_imp, tab_preds = st.tabs([
            "🎯 Actual vs Predicted",
            "📉 Residuals Diagnostics",
            "📊 Feature Importance",
            "📋 Holdout Predictions",
        ])

        with tab_scat:
            render_section_header("02", "Actual Ground Truth vs Model Prediction")
            if y_test is not None and y_pred is not None:
                st.plotly_chart(plot_actual_vs_predicted(y_test, y_pred), use_container_width=True)

        with tab_res:
            render_section_header("02", "Residual Error Diagnostics")
            if y_test is not None and y_pred is not None:
                st.plotly_chart(plot_residuals_chart(y_test, y_pred), use_container_width=True)

        with tab_imp:
            render_section_header("02", "Feature Impact & Magnitude")
            if imp_df is not None and not imp_df.empty:
                st.plotly_chart(plot_feature_importance_chart(imp_df), use_container_width=True)
                st.dataframe(imp_df.sort_values("Importance", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ Feature weights not available for this regressor.")

        with tab_preds:
            render_section_header("02", "Holdout Predictions & Error Table")
            if pred_df is not None:
                st.dataframe(pred_df, use_container_width=True)

    elif problem_type == "Clustering":
        labels = st.session_state.get("cluster_labels")
        X = get_active_data()
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist() if X is not None else []

        tab_dist, tab_pca3d, tab_data = st.tabs([
            "🧩 Cluster Distribution",
            "🌌 3D PCA Cluster Manifold",
            "📋 Clustered Dataset",
        ])

        with tab_dist:
            render_section_header("02", "Observation Density per Cluster")
            if labels is not None:
                c_counts = pd.Series(labels).value_counts().sort_index().reset_index()
                c_counts.columns = ["Cluster ID", "Samples Count"]
                st.dataframe(c_counts, use_container_width=True, hide_index=True)

        with tab_pca3d:
            render_section_header("02", "3D PCA Cluster Space Projection")
            if X is not None and len(num_cols) >= 3 and labels is not None:
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
            render_section_header("02", "Dataset with Assigned Cluster Labels")
            if X is not None and labels is not None:
                clustered_df = X.copy()
                clustered_df["Cluster_Label"] = labels
                st.dataframe(clustered_df.head(50), use_container_width=True)

    render_html("<br>")

    # =========================================================================
    # SECTION 3: UNIFIED TESTING & RAW PREDICTION HUB
    # =========================================================================
    render_section_header("03", "Live Model Testing & Raw Inference Hub")
    render_live_testing_hub(pipeline)

    # Model Hyperparameters Expander
    render_html("<br>")
    with st.expander("📦 Inspect Model Internal Hyperparameters & Architecture"):
        st.code(str(model), language="python")
        if hasattr(model, "get_params"):
            st.json(model.get_params())

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/11_Modeling.py",
        next_page="pages/14_Export.py",
        prev_label="← Model Studio",
        next_label="Export Center & Report →"
    )


if __name__ == "__main__":
    main()
