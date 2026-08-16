"""
Nexus AI — Page 01: Executive Dashboard & Command Center
--------------------------------------------------------
High-level telemetry, dataset health index, workflow progression matrix,
and instant stage navigation.
"""

import streamlit as st
import pandas as pd
from utils.session_state import init_session_state, get_active_data, get_dataset_telemetry, get_workflow_status, load_sample_dataset
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_step_navigation, render_sidebar_status, render_html


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    # Hero Banner
    render_hero(
        badge="● EXECUTIVE COMMAND CENTER",
        title="Nexus AI Analytics Hub",
        subtitle="End-to-end autonomous data intelligence, preprocessing pipeline telemetry, and machine learning orchestration."
    )

    df = get_active_data()
    t = get_dataset_telemetry(df)
    stages = get_workflow_status()
    done_count = sum(1 for s in stages if s["done"])
    progress_pct = int((done_count / len(stages)) * 100)

    # Top KPI Metrics Grid
    if df is not None:
        metrics = [
            {"icon": "🧬", "label": "Observations", "value": f"{t['rows']:,}", "sub": "Total Data Rows", "delta": "+Active", "delta_type": "pos"},
            {"icon": "📊", "label": "Features", "value": f"{t['cols']:,}", "sub": f"{t['numeric_count']} Num | {t['categorical_count']} Cat"},
            {"icon": "🩹", "label": "Missing Cells", "value": f"{t['missing_pct']:.1f}%", "sub": f"{t['missing_count']:,} Nulls", "delta_type": "neg" if t['missing_pct'] > 0 else "pos"},
            {"icon": "⚡", "label": "Pipeline Progress", "value": f"{progress_pct}%", "sub": f"{done_count}/{len(stages)} Stages Completed", "delta": f"Step {done_count+1}", "delta_type": "pos"},
        ]
    else:
        metrics = [
            {"icon": "📡", "label": "Connection", "value": "Awaiting", "sub": "No Dataset Connected"},
            {"icon": "📊", "label": "Features", "value": "0", "sub": "Ready for upload"},
            {"icon": "🩹", "label": "Quality Score", "value": "--", "sub": "Pending upload"},
            {"icon": "⚡", "label": "Pipeline Progress", "value": "0%", "sub": "Upload dataset to start"},
        ]

    render_metric_grid(metrics)

    # Quick Start or Active Dataset Card
    if df is None:
        with st.container(border=True):
            render_section_header("01", "Fast Start Ingestion")
            render_html(
                """
                <div style="color: #94A3B8; font-size: 14px; margin-bottom: 15px;">
                    No active dataset detected in memory. Upload your CSV/Excel file or choose an instant preloaded benchmark dataset.
                </div>
                """
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("🌸 Load Iris Flowers", use_container_width=True):
                    load_sample_dataset("iris")
                    st.rerun()
            with c2:
                if st.button("🚢 Load Titanic Survival", use_container_width=True):
                    load_sample_dataset("titanic")
                    st.rerun()
            with c3:
                if st.button("🍷 Load Wine Quality", use_container_width=True):
                    load_sample_dataset("wine")
                    st.rerun()
            with c4:
                if st.button("🏠 Load California Housing", use_container_width=True):
                    load_sample_dataset("california")
                    st.rerun()

    else:
        # Dataset Health & Telemetry Breakdown
        with st.container(border=True):
            render_section_header("01", "Dataset Health & Readiness Index")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Health Score", f"{t['health_score']}/100", delta="Ready" if t['health_score'] >= 80 else "Needs Cleaning")
            c2.metric("Duplicate Rows", f"{t['duplicate_count']:,}", delta=f"{t['duplicate_pct']:.1f}% of data", delta_color="inverse")
            c3.metric("Memory Footprint", f"{t['memory_mb']:.2f} MB")
            c4.metric("Active Model", st.session_state.get("model_name", "None"))

    # Interactive Pipeline Stages Grid
    render_html("<br>")
    with st.container(border=True):
        render_section_header("02", "Autonomous Workflow Architecture")
        render_html(
            """
            <div style="color: #94A3B8; font-size: 14px; margin-bottom: 18px;">
                Interactive status matrix across the 12 pipeline modules. Click any stage to jump directly into the workspace.
            </div>
            """
        )

        cols = st.columns(4)
        for idx, stage in enumerate(stages):
            col = cols[idx % 4]
            with col:
                status_icon = "✓ Done" if stage["done"] else "○ Pending"
                status_color = "#34D399" if stage["done"] else "#64748B"
                render_html(
                    f"""
                    <div style="background: rgba(15,23,42,0.6); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 14px; margin-bottom: 12px; transition: 0.2s;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size: 20px;">{stage['icon']}</span>
                            <span style="color: {status_color}; font-size: 11px; font-weight: 700; text-transform: uppercase;">{status_icon}</span>
                        </div>
                        <div style="color:#FFFFFF; font-weight:700; font-size:14px; margin-top:8px;">{stage['name']}</div>
                    </div>
                    """
                )
                if st.button(f"Open {stage['name']}", key=f"btn_jump_{stage['id']}", use_container_width=True, type="secondary"):
                    st.switch_page(stage["page"])

    # Bottom Step Navigation
    render_step_navigation(
        prev_page=None,
        next_page="pages/02_Upload.py",
        next_label="Begin Ingestion →"
    )


if __name__ == "__main__":
    main()
