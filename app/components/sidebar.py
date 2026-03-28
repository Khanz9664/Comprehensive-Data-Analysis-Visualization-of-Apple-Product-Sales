import streamlit as st
import pandas as pd

def render_sidebar(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    with st.sidebar:
        st.markdown("##  Analytics App")
        st.divider()
        
        # Navigation
        st.subheader("Navigation")
        page = st.radio(
            "Select View",
            options=["Executive Summary", "Overview", "Geospatial", "Analytics", "Segmentation", "Predictions", "Data Quality"],
            label_visibility="collapsed"
        )
        
        st.divider()
        st.subheader("Global Filters")
        
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
            label="⬇️ Export Filtered Data",
            data=csv_data,
            file_name="apple_sales_filtered_report.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )
        
        return filtered_df, page
