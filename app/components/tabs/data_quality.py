import streamlit as st
import pandas as pd
import numpy as np
from src.visualization.charts import plot_boxplot

def render(df: pd.DataFrame, data_quality: dict):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Quality Overview")
        st.write(f"**Total Records:** {data_quality['total_records']}")
        st.write(f"**Missing Values:** {data_quality['missing_values']}")
        st.write(f"**Duplicate Rows:** {data_quality['duplicate_rows']}")
        st.write(f"**Memory Usage:** {data_quality['memory_usage']:.2f} MB")

        schema_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(dtype) for dtype in df.dtypes],
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(schema_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Outlier Detection (IQR Method)")
        numeric_cols = df.select_dtypes(include=np.number).columns
        outlier_col = st.selectbox("Select Target Column", numeric_cols, key="outlier_select")
        
        Q1, Q3 = df[outlier_col].quantile(0.25), df[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[outlier_col] < (Q1 - 1.5 * IQR)) | (df[outlier_col] > (Q3 + 1.5 * IQR))]
        st.metric("Outliers Detected", len(outliers))
        
        fig_box = plot_boxplot(df, outlier_col)
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.divider()
    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.markdown("##### Business Logic")
        feature_cols = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales', 'Services_Revenue']
        if (df[feature_cols] < 0).any().any(): st.error("Negative Units Detected")
        else: st.success("Pass: No Negative Sales Values")

        if (df['Services_Revenue'] <= 0).any(): st.error("Invalid Revenue Values")
        else: st.success("Pass: All Revenue Positive")

    with vcol2:
        st.markdown("##### Consistency Mapping")
        expected = {'North America', 'Europe', 'Greater China', 'Rest of Asia', 'Rest of World'}
        if not set(df['Region'].unique()).issubset(expected): st.warning("Unexpected Regions Found")
        else: st.success("Pass: Region Schema Consistent")

        if df['Total_Product_Sales'].corr(df['Services_Revenue']) < 0.3: st.warning("Weak Product-Service Correlation")
        else: st.success("Pass: Correlation Baseline Met")
