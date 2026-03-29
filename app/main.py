import streamlit as st
import os

st.set_page_config(
    page_title="Apple Analytics Workspace",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_data, get_data_quality
from src.features.builder import build_features
from app.components.styles import apply_custom_css
from app.components.sidebar import render_sidebar
from app.components.kpi import render_kpis
from app.components.demo import initialize_demo, render_demo_overlay
from app.components.tabs import executive, overview, geospatial, analytics, segmentation, predictions, ai_assistant, data_quality, validation, export, live_mission_control

def main():
    apply_custom_css()
    initialize_demo()
    
    with st.spinner("Loading analytical models and data..."):
        df_raw = load_data('data/raw/apple_sales_2024.csv')
    
    if df_raw is None:
        st.error("Failed to initialize workspace data context.")
        st.stop()
        
    df = build_features(df_raw)
    dq_metrics = get_data_quality(df_raw)
    
    # Render Sidebar and Route
    filtered_df, page = render_sidebar(df)
    
    # Title
    st.title(f"Apple Sales Analytics / {page}")
    st.divider()
    
    # Executive KPIs
    render_kpis(filtered_df)
    
    # Execute Universal Presentation Overlay logic
    render_demo_overlay()
    
    # Conditional Page Routing
    if page == "Executive Summary":
        executive.render(filtered_df)
    elif page == "C-Suite AI Copilot":
        ai_assistant.render(filtered_df)
    elif page == "Overview":
        overview.render(filtered_df)
    elif page == "Geospatial":
        geospatial.render(filtered_df)
    elif page == "Analytics":
        analytics.render(filtered_df)
    elif page == "Advanced Segmentation":
        segmentation.render(filtered_df)
    elif page == "Predictions":
        predictions.render(filtered_df)
    elif page == "Data Quality":
        data_quality.render(filtered_df, dq_metrics)
    elif page == "Validation & Backtesting":
        validation.render(filtered_df)
    elif page == "Export & Reproducibility":
        export.render(filtered_df)
    elif page == "Live Mission Control":
        live_mission_control.render(filtered_df)

if __name__ == "__main__":
    main()
