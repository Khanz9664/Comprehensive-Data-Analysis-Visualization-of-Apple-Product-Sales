import streamlit as st
import pandas as pd
from src.analysis.statistics import perform_t_test, perform_anova, calculate_correlation
from src.visualization.charts import plot_scatter_with_trend, plot_distribution

def render(df: pd.DataFrame):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Independent T-Test")
        regions = df['Region'].unique()
        
        if len(regions) >= 2:
            r1 = st.selectbox("First Group", regions, key="r1")
            r2 = st.selectbox("Second Group", [r for r in regions if r != r1], key="r2")
            m_test = st.selectbox("Target Metric", ["iPhone_Sales", "Services_Revenue", "Total_Product_Sales"])
            
            t_stat, p_value, effect_size = perform_t_test(df, r1, r2, m_test)
            
            if t_stat is not None:
                ca, cb, cc = st.columns(3)
                ca.metric("T-Statistic", f"{t_stat:.3f}")
                cb.metric("P-Value", f"{p_value:.4f}")
                cc.metric("Cohen's d", f"{effect_size:.3f}")

    with col2:
        st.markdown("#### One-Way ANOVA")
        anova_m = st.selectbox("Target Variance Metric", ["iPhone_Sales", "iPad_Sales", "Mac_Sales", "Services_Revenue"])
        
        f_stat, p_value_anova = perform_anova(df, anova_m)
        if f_stat is not None:
            ca, cb = st.columns(2)
            ca.metric("F-Statistic", f"{f_stat:.3f}")
            cb.metric("P-Value", f"{p_value_anova:.4f}")
            
    st.divider()
    col3, col4 = st.columns(2)
    
    with col3:
        x_var = st.selectbox("X-Axis", ["iPhone_Sales", "iPad_Sales", "Mac_Sales", "Total_Product_Sales"])
        y_var = st.selectbox("Y-Axis", ["Services_Revenue", "Revenue_Per_Unit"])
        
        correlation, corr_p_value = calculate_correlation(df, x_var, y_var)
        fig_scatter = plot_scatter_with_trend(df, x_var, y_var, size_col='Total_Product_Sales')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with col4:
        dist_var = st.selectbox("Distribution Target", ["iPhone_Sales", "Services_Revenue", "Revenue_Per_Unit"])
        fig_dist = plot_distribution(df, dist_var)
        st.plotly_chart(fig_dist, use_container_width=True)
