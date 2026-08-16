"""
Nexus AI — Reusable UI Components & Navigation System
-----------------------------------------------------
High-fidelity visual components, Cyberpunk themes, KPI cards, Before/After matrices,
and unified workflow navigation.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from .session_state import get_workflow_status, get_dataset_telemetry, reset_workflow


def render_html(html_content: str):
    """
    Safely render raw HTML in Streamlit without Markdown interpreting
    indented lines as code blocks (<pre><code>).
    Uses native st.html if available, or unindented st.markdown fallback.
    """
    if hasattr(st, "html"):
        st.html(html_content)
    else:
        # Fallback for older Streamlit versions: strip leading whitespace from all lines
        lines = [line.strip() for line in html_content.strip().splitlines() if line.strip()]
        st.markdown("".join(lines), unsafe_allow_html=True)


def inject_global_css():
    """Inject the centralized Cyberpunk Glassmorphism CSS into the Streamlit app."""
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "styles.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        render_html(f"<style>{css_content}</style>")


def render_hero(badge: str, title: str, subtitle: str):
    """Render a standardized Cyberpunk glassmorphism hero banner."""
    html_content = f"""
    <div class="hero-container">
        <div class="hero-badge">{badge}</div>
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """
    render_html(html_content)


def render_section_header(number: str, title: str):
    """Render a cyber-styled section header."""
    html_content = f"""
    <div class="cyber-header">
        <span class="cyber-header-num">{number}</span>
        <span>{title}</span>
    </div>
    """
    render_html(html_content)


def render_metric_grid(metrics: list):
    """
    Render a responsive grid of glassmorphism KPI cards.
    Each item in metrics is a dict with: icon, label, value, sub (optional), delta (optional, pos/neg).
    """
    cards_html = []
    for m in metrics:
        icon = m.get("icon", "📊")
        label = m.get("label", "Metric")
        val = m.get("value", "--")
        sub = m.get("sub", "")
        delta = m.get("delta", None)
        delta_type = m.get("delta_type", "pos")

        delta_html = f'<span class="metric-badge-delta {delta_type}">{delta}</span>' if delta is not None else ""
        sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""

        card_html = f"""
        <div class="metric-card">
            <div class="metric-card-top">
                <div class="metric-icon">{icon}</div>
                {delta_html}
            </div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            {sub_html}
        </div>
        """
        cards_html.append(card_html)

    grid_html = f'<div class="metric-grid">{"".join(cards_html)}</div>'
    render_html(grid_html)


def render_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame, note: str = None):
    """Render a side-by-side Before/After KPI comparative panel."""
    t_before = get_dataset_telemetry(df_before)
    t_after = get_dataset_telemetry(df_after)

    row_delta = t_after["rows"] - t_before["rows"]
    col_delta = t_after["cols"] - t_before["cols"]
    miss_delta = t_after["missing_count"] - t_before["missing_count"]
    dup_delta = t_after["duplicate_count"] - t_before["duplicate_count"]

    def fmt_delta(val, invert_color=False):
        if val == 0:
            return '<span style="color:#94A3B8;">(0)</span>'
        elif val > 0:
            color = "#EF4444" if invert_color else "#10B981"
            return f'<span style="color:{color}; font-weight:700;">(+{val:,})</span>'
        else:
            color = "#10B981" if invert_color else "#EF4444"
            return f'<span style="color:{color}; font-weight:700;">({val:,})</span>'

    note_html = f'<div style="color:#38BDF8; font-size:13px; margin-top:10px; font-weight:600;">⚡ Note: {note}</div>' if note else ""

    comp_html = f"""
    <div class="comparison-container">
        <div class="comparison-box before">
            <div class="comparison-title">
                <span>⏮️</span> BEFORE TRANSFORMATION
            </div>
            <div class="comparison-row">
                <span class="comparison-label">Total Rows</span>
                <span class="comparison-val">{t_before['rows']:,}</span>
            </div>
            <div class="comparison-row">
                <span class="comparison-label">Total Features</span>
                <span class="comparison-val">{t_before['cols']:,}</span>
            </div>
            <div class="comparison-row">
                <span class="comparison-label">Missing Cells</span>
                <span class="comparison-val">{t_before['missing_count']:,} ({t_before['missing_pct']:.1f}%)</span>
            </div>
            <div class="comparison-row">
                <span class="comparison-label">Duplicate Rows</span>
                <span class="comparison-val">{t_before['duplicate_count']:,}</span>
            </div>
            <div class="comparison-row">
                <span class="comparison-label">Memory Footprint</span>
                <span class="comparison-val">{t_before['memory_mb']:.2f} MB</span>
            </div>
        </div>
        <div class="comparison-box after">
            <div class="comparison-title">
                <span>⏭️</span> AFTER TRANSFORMATION
            </div>
            <div class="comparison-row">
                <span class="comparison-label">Total Rows</span>
                <span class="comparison-val">{t_after['rows']:,} {fmt_delta(row_delta)}</span>
            </div>
            <div class="comparison-row">
                <span class="comparison-label">Total Features</span>
                <span class="comparison-val">{t_after['cols']:,} {fmt_delta(col_delta)}</span>
            </div>
            <div class="comparison-row">
                <span class="comparison-label">Missing Cells</span>
                <span class="comparison-val">{t_after['missing_count']:,} {fmt_delta(miss_delta, invert_color=True)}</span>
            </div>
            <div class="comparison-row">
                <span class="comparison-label">Duplicate Rows</span>
                <span class="comparison-val">{t_after['duplicate_count']:,} {fmt_delta(dup_delta, invert_color=True)}</span>
            </div>
            <div class="comparison-row">
                <span class="comparison-label">Memory Footprint</span>
                <span class="comparison-val">{t_after['memory_mb']:.2f} MB</span>
            </div>
        </div>
    </div>
    {note_html}
    """
    render_html(comp_html)


def render_step_navigation(prev_page: str = None, next_page: str = None, prev_label: str = "← Previous Step", next_label: str = "Next Step →"):
    """Render standard Previous and Next buttons at the bottom of the page."""
    render_html('<div class="nav-footer-container"></div>')
    col1, col2, col3 = st.columns([1.5, 3, 1.5])

    with col1:
        if prev_page:
            if st.button(prev_label, use_container_width=True, key="btn_nav_prev", type="secondary"):
                st.switch_page(prev_page)

    with col3:
        if next_page:
            if st.button(next_label, use_container_width=True, key="btn_nav_next"):
                st.switch_page(next_page)


def render_sidebar_status():
    """Render unified sidebar branding, dataset telemetry badge, and workflow stepper."""
    with st.sidebar:
        # Branding
        render_html("""
        <div class="sidebar-brand">
            <div class="sidebar-logo-text">⚡ NEXUS AI</div>
            <div class="sidebar-tagline">AI / ML Data Science Studio</div>
        </div>
        """)

        # Active Dataset Badge
        df = st.session_state.get("data")
        file_name = st.session_state.get("file_name", "No dataset uploaded")
        if df is not None:
            t = get_dataset_telemetry(df)
            render_html(f"""
            <div class="dataset-badge-card">
                <div class="dataset-badge-title">🟢 ACTIVE DATASET</div>
                <div class="dataset-badge-name" title="{file_name}">{file_name}</div>
                <div class="dataset-badge-stats">
                    <span>🧬 {t['rows']:,} rows</span>
                    <span>📊 {t['cols']} cols</span>
                    <span>🩹 {t['missing_pct']:.1f}% null</span>
                </div>
            </div>
            """)
        else:
            render_html("""
            <div class="dataset-badge-card" style="border-color: rgba(239,68,68,0.3);">
                <div class="dataset-badge-title" style="color:#EF4444;">🔴 NO DATASET</div>
                <div class="dataset-badge-name">Awaiting ingestion...</div>
                <div class="dataset-badge-stats"><span>Upload data to begin</span></div>
            </div>
            """)

        st.markdown("---")

        # Workflow Progress Indicator
        stages = get_workflow_status()
        done_count = sum(1 for s in stages if s["done"])
        progress_pct = int((done_count / len(stages)) * 100)

        render_html(f"""
        <div class="stepper-header">
            <span>PIPELINE PROGRESS</span>
            <span style="color:#38BDF8; font-weight:800;">{progress_pct}%</span>
        </div>
        """)
        st.progress(progress_pct / 100.0)

        # Quick Actions
        render_html("<br>")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if st.button("🔄 Reset", use_container_width=True, help="Reset transformations to original raw dataset"):
                reset_workflow(keep_original_file=True)
                st.rerun()
        with col_r2:
            if st.button("🗑️ Clear", use_container_width=True, help="Clear all data and start fresh"):
                reset_workflow(keep_original_file=False)
                st.switch_page("pages/02_Upload.py")

        st.caption("Nexus AI Engine v3.0 | Modern Connected Architecture")


def dataset_guard() -> bool:
    """
    Guard check for downstream pages. If no dataset is active, renders
    a user-friendly cyber message and stop button.
    Returns True if dataset exists, False if guarded (and stops execution).
    """
    if st.session_state.get("data") is None:
        render_hero(
            badge="⚠️ INGESTION REQUIRED",
            title="Dataset Uplink Required",
            subtitle="This stage requires an active dataset in memory. Please upload your CSV or Excel file or load a sample dataset."
        )
        st.warning("📡 No active dataset found in session state.")
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("📥 Jump to Dataset Ingestion", use_container_width=True):
                st.switch_page("pages/02_Upload.py")
        st.stop()
        return False
    return True
