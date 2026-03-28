import streamlit as st
import pandas as pd
from src.models.clustering import perform_clustering
from src.visualization.charts import plot_3d_clusters

def render(df: pd.DataFrame):
    n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=8, value=4)
    
    with st.spinner("Training KMeans Model..."):
        df_clustered, cluster_summary, silhouette = perform_clustering(df, n_clusters)
        
    st.metric("Silhouette Score", f"{silhouette:.3f}")

    x_col = st.selectbox("X-Axis", ['iPhone_Sales', 'Mac_Sales'])
    y_col = st.selectbox("Y-Axis", ['iPad_Sales', 'Wearables_Sales'])
    z_col = st.selectbox("Z-Axis", ['Services_Revenue', 'Total_Product_Sales'])
    size_col = st.selectbox("Dot Size", ['Revenue_Per_Unit', 'Total_Product_Sales'])

    fig_3d = plot_3d_clusters(df_clustered, x_col, y_col, z_col, size_col)
    st.plotly_chart(fig_3d, use_container_width=True, height=700)
    
    st.markdown("#### Cluster Centroids")
    st.dataframe(cluster_summary, use_container_width=True)
