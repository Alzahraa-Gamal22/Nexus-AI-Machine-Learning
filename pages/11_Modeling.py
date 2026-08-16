"""
Nexus AI — Page 11: Machine Learning Model Selection & Training Studio
----------------------------------------------------------------------
Multi-paradigm machine learning lab supporting Classification, Regression,
and Clustering with hyperparameter tuning and execution telemetry.
"""

import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error,
    silhouette_score
)
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from utils.session_state import init_session_state, get_active_data, get_dataset_telemetry
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_step_navigation, render_sidebar_status, dataset_guard, render_html


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    if not dataset_guard():
        return

    df = get_active_data()
    t = get_dataset_telemetry(df)

    render_hero(
        badge="● MACHINE LEARNING PROTOCOL",
        title="Model Selection & Training Studio",
        subtitle="Configure ML paradigms, tune algorithm hyperparameters, partition training sets, and execute high-performance training."
    )

    # 1. Paradigm & Target Configuration
    with st.container(border=True):
        render_section_header("01", "Learning Paradigm & Feature Architecture")

        col_p1, col_p2 = st.columns([1, 2])
        with col_p1:
            problem_options = ["Classification", "Regression", "Clustering"]
            prev_problem = st.session_state.get("problem_type", "Classification")
            def_idx = problem_options.index(prev_problem) if prev_problem in problem_options else 0
            problem_type = st.radio("Select ML Paradigm:", problem_options, index=def_idx, key="ml_problem_type")
            st.session_state.problem_type = problem_type

        with col_p2:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if problem_type in ["Classification", "Regression"]:
                target_options = df.columns.tolist()
                prev_target = st.session_state.get("target_column")
                def_t_idx = target_options.index(prev_target) if prev_target in target_options else len(target_options) - 1
                target_col = st.selectbox("🎯 Select Target Variable (Y):", target_options, index=def_t_idx, key="ml_target_col")
                st.session_state.target_column = target_col

                avail_features = [c for c in num_cols if c != target_col]
                selected_features = st.multiselect("Select Input Features (X):", avail_features, default=avail_features, key="ml_selected_features")
            else:
                st.session_state.target_column = None
                avail_features = num_cols
                selected_features = st.multiselect("Select Clustering Features (X):", avail_features, default=avail_features, key="ml_cluster_features")

    # Data Validation Guard
    if not selected_features:
        st.error("❌ Please select at least one numerical feature for modeling.")
        return

    X = df[selected_features].copy()
    y = df[target_col].copy() if problem_type != "Clustering" else None

    # Handle missing in X/y if any
    if X.isnull().any().any():
        st.error("❌ Missing values detected in modeling features. Please return to the Missing Values stage.")
        return
    if y is not None and y.isnull().any():
        st.error("❌ Missing values detected in the target column. Please resolve missing values before training.")
        return

    # 2. Model Architecture & Hyperparameters
    render_html("<br>")
    with st.container(border=True):
        render_section_header("02", "Algorithm Configuration & Hyperparameter Tuning")
        col_algo, col_params = st.columns([1, 2], gap="large")

        with col_algo:
            if problem_type == "Classification":
                algos = ["Random Forest", "Logistic Regression", "Support Vector Machine (SVM)", "Decision Tree", "K-Nearest Neighbors (KNN)", "Gaussian Naive Bayes", "Neural Network (MLP)"]
                if XGBOOST_AVAILABLE:
                    algos.append("XGBoost Classifier")
            elif problem_type == "Regression":
                algos = ["Linear Regression", "Ridge Regression", "Lasso Regression", "Decision Tree Regressor", "Random Forest Regressor", "Support Vector Regressor (SVR)"]
                if XGBOOST_AVAILABLE:
                    algos.append("XGBoost Regressor")
            else:
                algos = ["K-Means Clustering"]

            model_choice = st.selectbox("🤖 Select Algorithm:", algos, key="ml_algo_choice")

        with col_params:
            st.markdown("#### 🔧 Hyperparameters Matrix")
            model = None

            if "Random Forest" in model_choice:
                c1, c2, c3 = st.columns(3)
                n_est = c1.slider("Number of Trees:", 10, 500, 100, step=10)
                max_d = c2.slider("Max Depth:", 1, 50, 12)
                min_samp = c3.slider("Min Samples Split:", 2, 20, 2)
                if "Classifier" in model_choice or problem_type == "Classification":
                    model = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, min_samples_split=min_samp, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, min_samples_split=min_samp, random_state=42, n_jobs=-1)

            elif "Logistic Regression" in model_choice:
                c1, c2 = st.columns(2)
                c_val = c1.slider("Inverse Regularization (C):", 0.01, 10.0, 1.0)
                max_iter = c2.slider("Max Iterations:", 100, 3000, 1000, step=100)
                model = LogisticRegression(C=c_val, max_iter=max_iter, random_state=42)

            elif "Support Vector" in model_choice:
                c1, c2 = st.columns(2)
                c_val = c1.slider("Penalty Parameter (C):", 0.1, 10.0, 1.0)
                kernel = c2.selectbox("Kernel:", ["rbf", "linear", "poly"])
                if problem_type == "Classification":
                    model = SVC(C=c_val, kernel=kernel, probability=True, random_state=42)
                else:
                    model = SVR(C=c_val, kernel=kernel)

            elif "Decision Tree" in model_choice:
                c1, c2 = st.columns(2)
                max_d = c1.slider("Max Depth:", 1, 50, 10)
                min_leaf = c2.slider("Min Samples Leaf:", 1, 20, 1)
                if problem_type == "Classification":
                    model = DecisionTreeClassifier(max_depth=max_d, min_samples_leaf=min_leaf, random_state=42)
                else:
                    model = DecisionTreeRegressor(max_depth=max_d, min_samples_leaf=min_leaf, random_state=42)

            elif "K-Nearest" in model_choice:
                max_k = min(30, max(1, len(X) - 1))
                k_val = st.slider("Number of Neighbors (k):", 1, max_k, min(5, max_k))
                if problem_type == "Classification":
                    model = KNeighborsClassifier(n_neighbors=k_val)
                else:
                    model = KNeighborsRegressor(n_neighbors=k_val)

            elif "Naive Bayes" in model_choice:
                model = GaussianNB()
                st.info("Gaussian Naive Bayes operates on maximum likelihood Bayesian probabilities.")

            elif "Neural Network" in model_choice:
                c1, c2 = st.columns(2)
                hidden_nodes = c1.slider("Hidden Layer Neurons:", 10, 300, 100, step=10)
                epochs = c2.slider("Max Epochs / Iterations:", 100, 2000, 400, step=50)
                if problem_type == "Classification":
                    model = MLPClassifier(hidden_layer_sizes=(hidden_nodes,), max_iter=epochs, random_state=42)
                else:
                    model = MLPRegressor(hidden_layer_sizes=(hidden_nodes,), max_iter=epochs, random_state=42)

            elif "Linear Regression" in model_choice:
                model = LinearRegression()
                st.info("Ordinary Least Squares (OLS) Linear Regression.")

            elif "Ridge" in model_choice:
                alpha = st.slider("L2 Regularization Alpha:", 0.01, 100.0, 1.0)
                model = Ridge(alpha=alpha, random_state=42)

            elif "Lasso" in model_choice:
                alpha = st.slider("L1 Regularization Alpha:", 0.001, 10.0, 0.1)
                model = Lasso(alpha=alpha, random_state=42)

            elif "XGBoost" in model_choice:
                c1, c2 = st.columns(2)
                n_est = c1.slider("Boosting Trees:", 10, 500, 100, step=10)
                lr = c2.slider("Learning Rate:", 0.01, 0.5, 0.1, step=0.01)
                if "Classifier" in model_choice:
                    model = XGBClassifier(n_estimators=n_est, learning_rate=lr, random_state=42, eval_metric="logloss")
                else:
                    model = XGBRegressor(n_estimators=n_est, learning_rate=lr, random_state=42)

            elif "K-Means" in model_choice:
                k_val = st.slider("Number of Clusters (k):", 2, min(12, max(2, len(X)-1)), 3)
                model = KMeans(n_clusters=k_val, n_init=10, random_state=42)

    # 3. Partitioning & Execution
    render_html("<br>")
    with st.container(border=True):
        render_section_header("03", "Data Partitioning & Training Execution")
        
        test_pct = 0.20
        if problem_type in ["Classification", "Regression"]:
            test_pct = st.slider("Test Partition Ratio (Holdout Validation):", 0.10, 0.40, 0.20, step=0.05, format="%d%%")

        if st.button("⚡ INITIALIZE MODEL TRAINING", use_container_width=True, key="btn_train_model"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(100):
                    time.sleep(0.003)
                    progress_bar.progress(i + 1)
                    if i == 20:
                        status_text.info("🔄 Splitting data & partitioning feature matrices...")
                    elif i == 50:
                        status_text.info(f"🧠 Training {model_choice} architecture...")
                    elif i == 85:
                        status_text.info("📊 Calculating performance telemetry...")

                status_text.empty()
                progress_bar.empty()

                # Execution Logic
                if problem_type == "Classification":
                    # Stratify if minimum class count >= 2
                    class_counts = y.value_counts()
                    strat = y if class_counts.min() >= 2 else None
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_pct, random_state=42, stratify=strat)

                    t_start = time.time()
                    model.fit(X_tr, y_tr)
                    t_dur = time.time() - t_start
                    y_pred = model.predict(X_te)
                    y_prob = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None

                    acc = accuracy_score(y_te, y_pred)
                    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_te, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)

                    metrics_dict = {
                        "Accuracy": acc,
                        "Precision": prec,
                        "Recall": rec,
                        "F1 Score": f1,
                        "Training Latency": f"{t_dur:.3f}s",
                        "Train Samples": len(X_tr),
                        "Test Samples": len(X_te),
                    }

                    # Feature importance
                    imp_df = None
                    if hasattr(model, "feature_importances_"):
                        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
                    elif hasattr(model, "coef_"):
                        coefs = np.asarray(model.coef_)
                        imp_vals = np.abs(coefs) if coefs.ndim == 1 else np.mean(np.abs(coefs), axis=0)
                        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": imp_vals})

                    pred_df = pd.DataFrame({"Actual": y_te.reset_index(drop=True), "Predicted": pd.Series(y_pred).reset_index(drop=True)})
                    pred_df["Accurate"] = pred_df["Actual"] == pred_df["Predicted"]

                    st.session_state.update({
                        "model": model, "model_name": model_choice, "problem_type": problem_type,
                        "training_time": t_dur, "X_train": X_tr, "X_test": X_te, "y_train": y_tr,
                        "y_test": y_te, "y_pred": y_pred, "y_prob": y_prob,
                        "evaluation_metrics": metrics_dict, "feature_importance_df": imp_df,
                        "predictions_df": pred_df, "selected_features": selected_features,
                    })

                    st.success(f"✓ Model trained successfully! (Accuracy: {acc:.2%}, F1: {f1:.2%})")

                elif problem_type == "Regression":
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_pct, random_state=42)

                    t_start = time.time()
                    model.fit(X_tr, y_tr)
                    t_dur = time.time() - t_start
                    y_pred = model.predict(X_te)

                    r2 = r2_score(y_te, y_pred)
                    mae = mean_absolute_error(y_te, y_pred)
                    mse = mean_squared_error(y_te, y_pred)
                    rmse = np.sqrt(mse)

                    metrics_dict = {
                        "R² Score": r2,
                        "RMSE": rmse,
                        "MAE": mae,
                        "MSE": mse,
                        "Training Latency": f"{t_dur:.3f}s",
                        "Train Samples": len(X_tr),
                        "Test Samples": len(X_te),
                    }

                    imp_df = None
                    if hasattr(model, "feature_importances_"):
                        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
                    elif hasattr(model, "coef_"):
                        imp_df = pd.DataFrame({"Feature": X.columns, "Importance": np.abs(model.coef_)})

                    pred_df = pd.DataFrame({"Actual": y_te.reset_index(drop=True), "Predicted": pd.Series(y_pred).reset_index(drop=True)})
                    pred_df["Absolute Error"] = np.abs(pred_df["Actual"] - pred_df["Predicted"])

                    st.session_state.update({
                        "model": model, "model_name": model_choice, "problem_type": problem_type,
                        "training_time": t_dur, "X_train": X_tr, "X_test": X_te, "y_train": y_tr,
                        "y_test": y_te, "y_pred": y_pred, "evaluation_metrics": metrics_dict,
                        "feature_importance_df": imp_df, "predictions_df": pred_df,
                        "selected_features": selected_features,
                    })

                    st.success(f"✓ Regression model trained successfully! (R²: {r2:.4f}, RMSE: {rmse:.4f})")

                elif problem_type == "Clustering":
                    t_start = time.time()
                    labels = model.fit_predict(X)
                    t_dur = time.time() - t_start

                    sil = silhouette_score(X, labels) if 2 <= len(np.unique(labels)) < len(X) else 0.0

                    metrics_dict = {
                        "Number of Clusters": model.n_clusters,
                        "Silhouette Score": sil,
                        "Training Latency": f"{t_dur:.3f}s",
                        "Total Samples": len(X),
                    }

                    clustered_df = X.copy()
                    clustered_df["Cluster"] = labels

                    st.session_state.update({
                        "model": model, "model_name": model_choice, "problem_type": problem_type,
                        "training_time": t_dur, "cluster_labels": labels, "evaluation_metrics": metrics_dict,
                        "predictions_df": clustered_df, "selected_features": selected_features,
                    })

                    st.success(f"✓ Clustering model trained! (Silhouette Score: {sil:.4f})")

                st.rerun()

            except Exception as e:
                st.error(f"❌ Training failed: {e}")

    # Summary of Trained Model
    if st.session_state.get("model") is not None:
        render_html("<br>")
        with st.container(border=True):
            render_section_header("04", "Active Model Telemetry Snapshot")
            m = st.session_state.get("evaluation_metrics", {})
            m_cards = []
            for k, v in m.items():
                if isinstance(v, float):
                    val_str = f"{v:.4f}" if "Score" in k or "R²" in k else f"{v:.2%}" if "Accuracy" in k or "F1" in k else f"{v:.2f}"
                else:
                    val_str = str(v)
                m_cards.append({"icon": "⭐", "label": k, "value": val_str})
            render_metric_grid(m_cards[:4])

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/10_Visualization.py",
        next_page="pages/12_Evaluation.py",
        prev_label="← Visual Analytics",
        next_label="Deep Model Evaluation & Diagnostics →"
    )


if __name__ == "__main__":
    main()
