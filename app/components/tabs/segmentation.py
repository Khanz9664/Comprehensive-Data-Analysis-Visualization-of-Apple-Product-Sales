import streamlit as st
import pandas as pd
import plotly.express as px
from src.models.clustering import perform_advanced_clustering
from src.visualization.charts import plot_3d_clusters

def render(df: pd.DataFrame):
    if df.empty:
        st.warning("No data matches the current filters. Please adjust the sidebar settings.")
        return
        
    st.markdown("## Advanced Market Segmentation")
    st.write("Deconstruct global distribution patterns seamlessly mapping multi-dimensional pipeline distances into actionable strategic behavioral clusters.")
    
    st.divider()
    
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Clustering Algorithm", ["K-Means", "Agglomerative (Hierarchical)", "DBSCAN (Density-Based)"])
    model_key = method.split(" ")[0].lower()
    
    auto_k = False
    manual_k = 4
    eps = 0.5
    min_samples = 3
    
    # Conditional Layout Parameters tracking algorithmic requirements
    if model_key in ['k-means', 'agglomerative']:
        auto_k = c2.checkbox("Auto-Optimize Clusters (Silhouette & Elbow Method)", value=True)
        if not auto_k:
            manual_k = c3.slider("Manual Cluster Count", 2, 8, 4)
    else:
        eps = c2.slider("DBSCAN Epsilon (Radius)", 0.1, 2.0, 0.5, 0.1)
        min_samples = c3.slider("DBSCAN Min Samples", 2, 10, 3)
        
    model_key_param = 'kmeans' if model_key == 'k-means' else model_key
    
    with st.spinner("Executing Dimensional Distance Matrices..."):
        # Engage the refactored dynamic clustering block securely
        df_clustered, cluster_summary, silhouette, profiles, strategies, k_scores = perform_advanced_clustering(
            df, method=model_key_param, auto_k=auto_k, manual_k=manual_k, eps=eps, min_samples=min_samples
        )
        
    st.divider()
    
    # Metric Row Validation
    m1, m2 = st.columns(2)
    m1.metric("Silhouette Score (Convergence Quality)", f"{silhouette:.3f}")
    if auto_k and k_scores:
        m2.metric("Optimal Centers Extracted", f"{len(profiles)}")
        
    # Render elbow trajectory natively over Plotly
    if auto_k and k_scores:
        score_df = pd.DataFrame(list(k_scores.items()), columns=['K', 'Silhouette Score'])
        fig_scores = px.line(score_df, x='K', y='Silhouette Score', markers=True, title="Algorithm Processing Trajectory (Silhouettes)")
        fig_scores.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig_scores, use_container_width=True)

    st.divider()
    
    with st.container(border=True):
        st.markdown("### Interactive 3D Architecture")
        col_x = st.selectbox("X-Axis Segment", ['iPhone_Sales', 'Services_Revenue', 'Wearables_Sales'], key='x')
        col_y = st.selectbox("Y-Axis Segment", ['iPad_Sales', 'Mac_Sales', 'Total_Product_Sales'], key='y')
        col_z = st.selectbox("Z-Axis Segment", ['Services_Revenue', 'iPhone_Sales', 'Revenue_Per_Unit'], index=0, key='z')
        
        try:
            if 'Total_Product_Sales' not in df_clustered.columns:
                df_clustered['Total_Product_Sales'] = df_clustered[['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']].sum(axis=1)
            fig_3d = plot_3d_clusters(df_clustered, col_x, col_y, col_z, size_col='Total_Product_Sales')
            fig_3d.update_layout(paper_bgcolor="#1A1F2B")
            st.plotly_chart(fig_3d, use_container_width=True, height=750)
        except Exception as e:
            st.error(f"Fallback rendering engaged natively: {str(e)}")

    st.divider()

    with st.container(border=True):
        st.markdown("### Strategic Cluster Deployments (Dynamic Profiling)")
        st.write("Unsupervised matrices algorithmically evaluated the exact topological centroids of your active clusters to dynamically write out the semantic definitions below:")
        
        # Print dynamic profiles and strategies natively securely handling arrays
        for cluster_id, profile_label in profiles.items():
            if cluster_id == -1:
                st.error(f"**{profile_label}:** {strategies.get(cluster_id, '')}")
            else:
                strat = strategies.get(cluster_id, "")
                st.success(f"**{profile_label}**\n\n**Actionable Logic:** {strat}")
