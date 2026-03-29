import streamlit as st
import pandas as pd
from app.components.demo import start_demo

def render_sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    with st.sidebar:
        # Custom Sidebar Styling
        st.markdown("<h2 style='text-align: center;'> Executive Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Machine Learning Division</p>", unsafe_allow_html=True)
        st.divider()
        
        # Navigation
        st.subheader("Step-by-Step Workflow")
        
        if 'nav_radio' not in st.session_state:
            st.session_state.nav_radio = "Live Mission Control"
            
        options_list = [
            "Live Mission Control",
            "Executive Summary",
            "Overview",
            "Data Quality",
            "Geospatial",
            "Analytics",
            "Advanced Segmentation",
            "Predictions",
            "Validation & Backtesting",
            "Export & Reproducibility",
            "C-Suite AI Copilot"
        ]

        st.radio(
            "Select View",
            options=options_list,
            key="nav_radio",
            label_visibility="collapsed",
            disabled=st.session_state.get('demo_active', False)
        )
        page = st.session_state.nav_radio
        
        st.divider()
        st.subheader("Global Control Array")
        
        if not st.session_state.get('demo_active', False):
            st.button("Launch Guided Demo", on_click=start_demo, type="primary", use_container_width=True)
            st.divider()
        else:
            st.warning("**Guided Pitch Active:** Standard navigation layouts are temporarily locked by the presentation logic.")
        
        # Product Category Filter (Simulated via multiselect on numeric columns by selecting non-zero rows, or just standard Region filter)
        selected_regions = st.multiselect(
            "Region",
            options=df['Region'].unique(),
            default=df['Region'].unique()
        )
        
        sales_threshold = st.slider(
            "Min iPhone Sales (M)",
            min_value=float(df['iPhone_Sales'].min()),
            max_value=float(df['iPhone_Sales'].max()),
            value=float(df['iPhone_Sales'].min()),
            step=1.0
        )
        
        filtered_df = df[
            (df['Region'].isin(selected_regions)) & 
            (df['iPhone_Sales'] >= sales_threshold)
        ]
        
        st.divider()
        
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Filtered Data",
            data=csv_data,
            file_name="apple_sales_filtered_report.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
        
        return filtered_df, page
