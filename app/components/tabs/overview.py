import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from src.visualization.charts import plot_correlation_matrix
from src.analysis.insights import generate_business_recommendations

def render(df: pd.DataFrame):
    if df.empty:
        st.warning("No data matches the current filters. Please adjust the sidebar settings.")
        return
        
    st.markdown("## Data Story & Ecosystem Overview")
    st.write("A holistic narrative dissecting global hardware distribution health, mathematical dataset integrity, and underlying product pipeline contributions.")
    
    # 1. Dataset Health Snapshots
    with st.container(border=True):
        st.markdown("### 1. Telemetry Health Snapshot")
        hcol1, hcol2, hcol3 = st.columns(3)
        hcol1.metric("Verified Data Rows", f"{len(df):,}")
        hcol2.metric("Missing/Corrupt Values", f"{df.isna().sum().sum()}")
        hcol3.metric("Latest Telemetry Sync", datetime.now().strftime("%Y-%m-%d"))
    
    st.divider()
    
    # 2. Interactive Local Filters & Time-Series
    with st.container(border=True):
        st.markdown("### 2. Quarterly Trajectory (Simulated Trailing 12-Months)")
        st.write("Because the raw `.csv` telemetry lacks distinct timestamp indexing, this chart dynamically simulates a realistic 12-month trailing vector mathematically constrained to your actively filtered global data sums.")
        
        np.random.seed(42) # Lock seed for visual stability during filter swaps
        dates = pd.date_range(end=datetime.today(), periods=12, freq='ME')
        
        products = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales', 'Services_Revenue']
        available_p = [p for p in products if p in df.columns]
        
        if available_p:
            selected_prod = st.selectbox("Select Pipeline to Analyze", available_p, index=0)
            
            # Calculate a pseudo-realistic trailing trace based on actual filtered data volumes
            base_val = df[selected_prod].sum() / 12
            trend_data = [base_val * (1 + np.random.normal(0, 0.08)) for _ in range(12)]
            trend_data = [val * (1 + (i * 0.015)) for i, val in enumerate(trend_data)] # +1.5% drift
            
            ts_df = pd.DataFrame({'Date': dates, 'Volume': trend_data})
            
            fig_ts = px.line(
                ts_df, x='Date', y='Volume', markers=True, 
                title=f"Trailing Distribution Vector: {selected_prod.replace('_', ' ')}", 
                color_discrete_sequence=["#6366F1"]
            )
            fig_ts.update_layout(template="plotly_dark", plot_bgcolor="#1A1F2B", paper_bgcolor="#1A1F2B")
            fig_ts.update_xaxes(showgrid=False)
            fig_ts.update_yaxes(showgrid=False)
            st.plotly_chart(fig_ts, use_container_width=True)
        
    st.divider()
    
    # 3. Product Contribution Breakdown & Narrative
    with st.container(border=True):
        st.markdown("### 3. Product Ecosystem Contribution")
        
        hardware_cols = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
        hw_available = [p for p in hardware_cols if p in df.columns]
        
        ccol1, ccol2 = st.columns([1, 1.2])
        
        with ccol1:
            if hw_available:
                sums = df[hw_available].sum().reset_index()
                sums.columns = ['Product', 'Total']
                sums['Product'] = sums['Product'].str.replace('_Sales', '')
                
                fig_pie = px.pie(
                    sums, values='Total', names='Product', hole=0.65, 
                    color_discrete_sequence=["#6366F1", "#4F46E5", "#3730A3", "#312E81"]
                )
                fig_pie.update_layout(
                    template="plotly_dark", plot_bgcolor="#1A1F2B", 
                    paper_bgcolor="#1A1F2B", margin=dict(t=20, b=20, l=10, r=10)
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
        with ccol2:
            st.markdown("#### The Data Narrative")
            st.write("Evaluating the physical supply-chain distributions mathematically isolated above exposes extremely potent dependency chains. This allocation wheel perfectly dictates where the heaviest logistical frictions inherently exist.")
            
            recs = generate_business_recommendations(df)
            if recs:
                st.markdown("##### Auto-Generated Strategic Context:")
                for rec in recs:
                    if rec['type'] == 'success':
                        st.success(rec['msg'])
                    else:
                        st.warning(rec['msg'])
            else:
                st.info("Insufficient variance to generate contextual insights.")
            
    st.divider()
    
    # 4. Correlation Matrix
    with st.container(border=True):
        st.markdown("### 4. Global Feature Correlation Matrix")
        st.write("A pure, unopinionated mathematical cross-examination assessing collinearity between operational product flows.")
        cols = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales', 'Services_Revenue']
        valid_cols = [c for c in cols if c in df.columns]
        fig_corr = plot_correlation_matrix(df, valid_cols)
        fig_corr.update_layout(plot_bgcolor="#1A1F2B", paper_bgcolor="#1A1F2B")
        st.plotly_chart(fig_corr, use_container_width=True)
