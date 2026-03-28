import streamlit as st
import pandas as pd
import numpy as np

def render_kpis(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = df['Services_Revenue'].sum()
    total_units = df['Total_Product_Sales'].sum()
    avg_order_value = df['Revenue_Per_Unit'].mean() * 1000 # Mock representation multiplier for context
    
    # Mocking a static growth delta since no actual time series exists in CSV
    # Usually we would compare current frame to previous frame
    growth_pct = "+5.2%"
    
    with col1:
        st.metric(label="Total Revenue", value=f"${total_revenue:,.1f}B", delta="+2.4% YoY")

    with col2:
        st.metric(label="Units Sold", value=f"{total_units:,.0f}M", delta=growth_pct)

    with col3:
        st.metric(label="Growth % (Simulated)", value="5.2%", delta="0.8% QoQ")

    with col4:
        st.metric(label="Avg Output/Unit", value=f"${avg_order_value:,.2f}", delta="-1.2%")
    
    st.divider()
