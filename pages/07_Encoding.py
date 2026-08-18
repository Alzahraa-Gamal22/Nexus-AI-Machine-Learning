"""
Nexus AI — Page 07: Categorical Encoding Studio
-----------------------------------------------
Intelligent vectorization of non-numeric features using One-Hot Encoding,
Label Encoding, Ordinal Encoding, and Frequency Encoding.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.session_state import init_session_state, get_active_data, set_active_data, get_dataset_telemetry
from utils.ui import inject_global_css, render_hero, render_section_header, render_metric_grid, render_before_after, render_step_navigation, render_sidebar_status, dataset_guard, render_html
from utils.preprocessing import apply_categorical_encoding


def main():
    init_session_state()
    inject_global_css()
    render_sidebar_status()

    if not dataset_guard():
        return

    df = get_active_data()
    t = get_dataset_telemetry(df)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    render_hero(
        badge="● VECTORIZATION PROTOCOL",
        title="Categorical Encoding Studio",
        subtitle="Transform non-numerical features into mathematical vectors tailored for machine learning models."
    )

    if not cat_cols:
        st.success("✓ All features in the active dataset are already numerical! No encoding required.")
        render_step_navigation(prev_page="pages/06_Outliers.py", next_page="pages/08_Feature_Engineering.py")
        return

    # Top KPI Metrics
    high_card_count = sum(df[c].nunique() > 15 for c in cat_cols)
    binary_count = sum(df[c].nunique() == 2 for c in cat_cols)

    metrics = [
        {"icon": "🏷️", "label": "Categorical Features", "value": f"{len(cat_cols)}", "sub": f"Out of {t['cols']} features"},
        {"icon": "⚖️", "label": "Binary Features", "value": f"{binary_count}", "sub": "Exactly 2 classes"},
        {"icon": "⚠️", "label": "High Cardinality", "value": f"{high_card_count}", "sub": ">15 unique classes", "delta_type": "neg" if high_card_count > 0 else "pos"},
        {"icon": "⚡", "label": "Encoding Status", "value": "Encoded" if st.session_state.get("encoding_done") else "Pending", "sub": "Pipeline stage 07", "delta_type": "pos"},
    ]
    render_metric_grid(metrics)

    # Encoding Strategy Matrix
    with st.container(border=True):
        render_section_header("01", "Feature Encoding Configuration Matrix")
        render_html(
            """
            <div style="color: #94A3B8; font-size: 14px; margin-bottom: 15px;">
                Review automated recommendations or manually configure vectorization strategies for each feature:
            </div>
            """
        )

        decisions = {}
        cols_grid = st.columns(min(3, max(1, len(cat_cols))))

        for idx, col in enumerate(cat_cols):
            c = cols_grid[idx % len(cols_grid)]
            n_uniq = df[col].nunique()

            # Smart Default
            if n_uniq <= 2:
                rec = "Label"
            elif n_uniq <= 10:
                rec = "One-Hot"
            else:
                rec = "Frequency"

            with c:
                st.markdown(f"**{col}** (`{n_uniq}` unique values)")
                options = ["One-Hot", "Label", "Ordinal", "Frequency"]
                default_idx = options.index(rec)
                choice = st.selectbox(
                    f"Strategy for {col}:",
                    options,
                    index=default_idx,
                    key=f"enc_strat_{col}",
                    help=f"Recommended: {rec} based on cardinality of {n_uniq}."
                )

                strategy_key = "one_hot"
                if choice == "Label":
                    strategy_key = "label"
                elif choice == "Ordinal":
                    strategy_key = "ordinal"
                elif choice == "Frequency":
                    strategy_key = "frequency"

                decisions[col] = strategy_key

        render_html("<br>")
        if st.button("⚡ EXECUTE ENCODING PIPELINE", use_container_width=True, key="btn_apply_encoding"):
            df_encoded, encoders_dict = apply_categorical_encoding(df, decisions)
            st.session_state.encoding_done = True
            if st.session_state.get("preprocessing_pipeline"):
                st.session_state.preprocessing_pipeline.record_encoding(decisions, encoders_dict)
            set_active_data(df_encoded, stage_name="Encoding", log_entry=f"Encoded {len(decisions)} categorical features: {decisions}.")
            st.success(f"✓ Successfully encoded {len(decisions)} categorical features into {df_encoded.shape[1]} numeric columns!")
            st.rerun()

    # Transformed Preview & Before/After
    orig_df = st.session_state.get("original_data")
    if orig_df is not None:
        render_html("<br>")
        render_section_header("02", "Vectorization Audit Telemetry")
        render_before_after(orig_df, df, note="Monitoring expansion and transformation of categorical dimensions.")

    # Navigation Footer
    render_step_navigation(
        prev_page="pages/06_Outliers.py",
        next_page="pages/08_Feature_Engineering.py",
        prev_label="← Outlier Treatment",
        next_label="Feature Engineering & Selection →"
    )


if __name__ == "__main__":
    main()
