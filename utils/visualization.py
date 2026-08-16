"""
Nexus AI — Plotly Visualization Engine
--------------------------------------
High-contrast, interactive 2D & 3D visualization studio and ML diagnostic charts.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def _apply_cyber_theme(fig: go.Figure, title: str = None) -> go.Figure:
    """Apply uniform cyberpunk dark glassmorphism styling to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Outfit, sans-serif", size=16, color="#F8FAFC"),
            x=0.02,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(family="Outfit, sans-serif", color="#CBD5E1"),
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(
            bgcolor="rgba(15,23,42,0.6)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


# ============================================================
# 1. 2D EXPLORATORY VISUALIZATIONS
# ============================================================

def plot_histogram(df: pd.DataFrame, x_col: str, color_col: str = None, n_bins: int = 30, kde: bool = False) -> go.Figure:
    """Generate distribution histogram with optional color segmentation and marginal box."""
    marginal = "box" if kde else None
    fig = px.histogram(
        df,
        x=x_col,
        color=color_col if color_col and color_col != "None" else None,
        nbins=n_bins,
        marginal=marginal,
        color_discrete_sequence=["#38BDF8", "#8B5CF6", "#EC4899", "#10B981", "#F59E0B"],
        opacity=0.85,
    )
    return _apply_cyber_theme(fig, f"Distribution of {x_col}")


def plot_boxplot(df: pd.DataFrame, y_col: str, x_col: str = None, color_col: str = None) -> go.Figure:
    """Generate box / violin plot."""
    x_val = x_col if x_col and x_col != "None" else None
    c_val = color_col if color_col and color_col != "None" else None
    fig = px.box(
        df,
        y=y_col,
        x=x_val,
        color=c_val,
        points="outliers",
        color_discrete_sequence=["#8B5CF6", "#38BDF8", "#EC4899", "#34D399"],
    )
    return _apply_cyber_theme(fig, f"Box Plot: {y_col}" + (f" by {x_col}" if x_val else ""))


def plot_scatter_2d(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, size_col: str = None, trendline: bool = False) -> go.Figure:
    """Generate 2D scatter plot with optional trendline and groupings."""
    c_val = color_col if color_col and color_col != "None" else None
    s_val = size_col if size_col and size_col != "None" else None
    t_val = "ols" if trendline else None

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=c_val,
        size=s_val,
        trendline=t_val,
        color_discrete_sequence=["#38BDF8", "#EC4899", "#8B5CF6", "#10B981"],
        opacity=0.8,
    )
    return _apply_cyber_theme(fig, f"{y_col} vs {x_col}")


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list = None) -> go.Figure:
    """Generate interactive correlation heatmap with annotated coefficients."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        fig = go.Figure()
        fig.add_annotation(text="At least 2 numerical features are required for correlation analysis.", showarrow=False)
        return _apply_cyber_theme(fig, "Correlation Matrix")

    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="Purples",
        aspect="auto",
        labels=dict(color="Correlation"),
    )
    return _apply_cyber_theme(fig, "Feature Correlation Heatmap")


def plot_missing_matrix(df: pd.DataFrame) -> go.Figure:
    """Visual heatmap of missing values across all rows and columns."""
    sample_df = df.sample(min(len(df), 500), random_state=42) if len(df) > 500 else df
    null_matrix = sample_df.isnull().astype(int)

    fig = px.imshow(
        null_matrix,
        color_continuous_scale=[[0, "#0F172A"], [1, "#EF4444"]],
        labels=dict(x="Features", y="Row Index", color="Missing"),
        aspect="auto",
    )
    return _apply_cyber_theme(fig, "Missing Values Heatmap (1 = Missing, 0 = Present)")


def plot_categorical_distribution(df: pd.DataFrame, col: str, chart_type: str = "bar") -> go.Figure:
    """Bar or Donut chart showing categorical class distribution."""
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]

    if chart_type == "donut":
        fig = px.pie(
            counts,
            names=col,
            values="Count",
            hole=0.5,
            color_discrete_sequence=px.colors.qualitative.Prism,
        )
    else:
        fig = px.bar(
            counts,
            x=col,
            y="Count",
            text="Count",
            color=col,
            color_discrete_sequence=["#38BDF8", "#8B5CF6", "#EC4899", "#10B981", "#F59E0B"],
        )
    return _apply_cyber_theme(fig, f"Distribution of {col}")


# ============================================================
# 2. 3D INTERACTIVE VISUALIZATIONS
# ============================================================

def plot_scatter_3d(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, color_col: str = None, size_col: str = None) -> go.Figure:
    """Generate high-end 3D scatter plot."""
    c_val = color_col if color_col and color_col != "None" else None
    s_val = size_col if size_col and size_col != "None" else None

    fig = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color=c_val,
        size=s_val,
        color_discrete_sequence=["#38BDF8", "#8B5CF6", "#EC4899", "#10B981", "#F59E0B"],
        opacity=0.85,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(15,23,42,0.5)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(backgroundcolor="rgba(15,23,42,0.5)", gridcolor="rgba(255,255,255,0.08)"),
            zaxis=dict(backgroundcolor="rgba(15,23,42,0.5)", gridcolor="rgba(255,255,255,0.08)"),
        ),
        title=dict(text=f"<b>3D Feature Space: {x_col} × {y_col} × {z_col}</b>", font=dict(family="Outfit", size=16)),
    )
    return fig


def plot_3d_pca_space(pca_df: pd.DataFrame, color_col: str = None, explained_variance: list = None) -> go.Figure:
    """Render 3D PCA projection space."""
    c_val = color_col if color_col and color_col in pca_df.columns else None

    var_str = ""
    if explained_variance is not None and len(explained_variance) >= 3:
        var_str = f" (PC1: {explained_variance[0]*100:.1f}%, PC2: {explained_variance[1]*100:.1f}%, PC3: {explained_variance[2]*100:.1f}%)"

    fig = px.scatter_3d(
        pca_df,
        x="PC_1",
        y="PC_2",
        z="PC_3",
        color=c_val,
        color_discrete_sequence=["#38BDF8", "#EC4899", "#8B5CF6", "#10B981"],
        opacity=0.85,
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(15,23,42,0.5)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(backgroundcolor="rgba(15,23,42,0.5)", gridcolor="rgba(255,255,255,0.08)"),
            zaxis=dict(backgroundcolor="rgba(15,23,42,0.5)", gridcolor="rgba(255,255,255,0.08)"),
        ),
        title=dict(text=f"<b>3D Principal Component Space{var_str}</b>", font=dict(family="Outfit", size=16)),
    )
    return fig


# ============================================================
# 3. MACHINE LEARNING DIAGNOSTIC VISUALIZATIONS
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, labels: list) -> go.Figure:
    """Plot interactive confusion matrix heatmap."""
    fig = px.imshow(
        cm,
        text_auto=True,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        color_continuous_scale="Purples",
    )
    return _apply_cyber_theme(fig, "Confusion Matrix")


def plot_roc_curve_chart(y_test, y_probs) -> go.Figure:
    """Generate ROC-AUC curve."""
    fig = go.Figure()

    if y_probs.ndim == 1 or y_probs.shape[1] == 2:
        probs = y_probs[:, 1] if y_probs.ndim > 1 else y_probs
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC = {roc_auc:.3f})", line=dict(color="#38BDF8", width=3)))
    else:
        # Multi-class
        for i in range(y_probs.shape[1]):
            binary_y = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_y, y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"Class {i} (AUC = {roc_auc:.3f})"))

    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(color="rgba(255,255,255,0.3)", dash="dash")))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return _apply_cyber_theme(fig, "Receiver Operating Characteristic (ROC) Curve")


def plot_precision_recall_chart(y_test, y_probs) -> go.Figure:
    """Generate Precision-Recall curve."""
    fig = go.Figure()
    probs = y_probs[:, 1] if y_probs.ndim > 1 else y_probs
    precision, recall, _ = precision_recall_curve(y_test, probs)
    fig.add_trace(go.Scatter(x=recall, y=precision, name="PR Curve", line=dict(color="#8B5CF6", width=3), fill="tozeroy"))
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return _apply_cyber_theme(fig, "Precision-Recall Curve")


def plot_feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    sorted_df = importance_df.sort_values("Importance", ascending=True)
    fig = px.bar(
        sorted_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Tealgrn",
    )
    return _apply_cyber_theme(fig, "Feature Importance Telemetry")


def plot_actual_vs_predicted(y_test, y_pred) -> go.Figure:
    """Scatter plot of actual vs predicted with ideal diagonal line for regression."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="Predictions", marker=dict(color="#38BDF8", opacity=0.75, size=7)))

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="Ideal 1:1 Line", line=dict(color="#EC4899", dash="dash", width=2)))

    fig.update_xaxes(title="Actual Ground Truth")
    fig.update_yaxes(title="Model Predicted Value")
    return _apply_cyber_theme(fig, "Actual vs Predicted Values")


def plot_residuals_chart(y_test, y_pred) -> go.Figure:
    """Plot regression residuals distribution and vs-fitted."""
    residuals = y_test - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers", marker=dict(color="#8B5CF6", opacity=0.75, size=7)))
    fig.add_hline(y=0, line_dash="dash", line_color="#10B981")
    fig.update_xaxes(title="Fitted / Predicted Values")
    fig.update_yaxes(title="Residuals (Error)")
    return _apply_cyber_theme(fig, "Residuals vs Fitted Values")
