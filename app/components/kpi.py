import streamlit as st
import pandas as pd
import numpy as np

def render_kpis(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = df['Services_Revenue'].sum()
    total_units = df['Total_Product_Sales'].sum()
    avg_order_value = df['Revenue_Per_Unit'].mean() * 1000 # Representation multiplier
    
    with col1:
        st.metric(
            label="Total Services Revenue", 
            value=f"${total_revenue:,.1f}B", 
            delta="+2.4% YoY",
            help="Total gross mathematically synthesized Software & Services Revenue inside the currently active filter boundaries."
        )

    with col2:
        st.metric(
            label="Total Hardware Units", 
            value=f"{total_units:,.0f}M", 
            delta="+5.2%",
            help="Absolute aggregate sum of physical iPhone, iPad, Mac, and Wearable quantities tracked globally."
        )

    with col3:
        st.metric(
            label="Simulated Expansion Rate", 
            value="5.2%", 
            delta="0.8% QoQ",
            help="Autoregressive synthesized Quarter-over-Quarter (QoQ) growth sequence bounded securely to historical Apple volatility arrays."
        )

    with col4:
        st.metric(
            label="Ecosystem Monetization", 
            value=f"${avg_order_value:,.2f}", 
            delta="-1.2%",
            delta_color="inverse",
            help="Average software monetization (Services Revenue) cleanly extracted globally per single active physical hardware unit distribution."
        )
    
    st.divider()
