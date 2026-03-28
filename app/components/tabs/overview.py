import streamlit as st
import pandas as pd
from src.visualization.charts import plot_correlation_matrix, plot_radar_chart
from src.analysis.insights import generate_business_recommendations

def render(df: pd.DataFrame):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        cols_to_corr = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales', 'Services_Revenue']
        fig_corr = plot_correlation_matrix(df, cols_to_corr)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col2:
        fig_radar = plot_radar_chart(df)
        st.plotly_chart(fig_radar, use_container_width=True)
        
    st.divider()
    st.markdown("### Strategic Recommendations")
    
    recommendations = generate_business_recommendations(df)
    
    if not recommendations:
        st.info("Insufficient data to generate strategic recommendations.")
    else:
        for rec in recommendations:
            if rec['type'] == 'success':
                st.success(rec['msg'])
            else:
                st.warning(rec['msg'])
