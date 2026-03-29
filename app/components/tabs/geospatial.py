import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from src.models.regression import get_production_pipeline

def render(df: pd.DataFrame):
    if df.empty:
        st.warning("No data matches the current filters. Please adjust the sidebar settings.")
        return
        
    st.markdown("## Geospatial Intelligence & Expansion Strategy")
    st.write("Transitioning from pure geographical mapping into predictive regional forecasting, saturation clustering, and hierarchical density layouts.")
    
    st.divider()
    
    # Ensure necessary columns
    req_cols = ['Region', 'State', 'Services_Revenue', 'iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
    available_cols = [c for c in req_cols if c in df.columns]
    
    if len(available_cols) < len(req_cols):
        st.error("Insufficient dataset metrics to run geospatial analytics.")
        return
        
    df = df.copy() # Safe dataframe isolation
    df['Total_Hardware'] = df[['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']].sum(axis=1)

    # 1. Market Tiers Clustering (K-Means)
    with st.container(border=True):
        st.markdown("### 1. Global Market Clustering (Density & Monetization)")
        
        # Cluster strictly by Hardware Volume vs Software Monetization
        cluster_data = df[['Total_Hardware', 'Services_Revenue']].fillna(0)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Market_Tier'] = kmeans.fit_predict(cluster_data)
        
        # Rename clusters based on centroids to ensure Tier 1 is highest revenue
        cluster_means = df.groupby('Market_Tier')['Services_Revenue'].mean().sort_values(ascending=False)
        tier_map = {cluster_means.index[0]: 'Tier 1 (Flagship)', cluster_means.index[1]: 'Tier 2 (Emerging)', cluster_means.index[2]: 'Tier 3 (Untapped)'}
        df['Market_Tier_Label'] = df['Market_Tier'].map(tier_map)
        
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            fig_scatter = px.scatter(
                df, x='Total_Hardware', y='Services_Revenue', 
                color='Market_Tier_Label', hover_name='State',
                title="Market Saturation vs Growth Potential",
                color_discrete_sequence=["#00B4D8", "#03045E", "#90E0EF"],
                labels={'Total_Hardware': 'Total Hardware Units (M)'}
            )
            fig_scatter.update_layout(template="plotly_dark", plot_bgcolor="#1A1F2B", paper_bgcolor="#1A1F2B", margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with col2:
            st.markdown("#### Strategic Posture")
            st.write("We deployed an Unsupervised Machine Learning algorithm (`K-Means`) across all global districts to instantly isolate **Flagship Markets** (High Hardware, High Revenue) entirely from **Untapped Sectors**.")
            st.info("**Action Requirement:** Aggressively deploy localized Apple One bundling promotions strictly into the *Tier 2* and *Tier 3* clusters. The goal is to force the scatter trajectory vertically on the Y-Axis into the Flagship zone.")

    st.divider()

    # 2. Revenue Density & Hierarchical Drill-Downs
    with st.container(border=True):
        st.markdown("### 2. Global Density Drill-Down Engine")
        st.write("Click any Region block below to seamlessly drill down mathematically into specific State/Country level market distributions.")
        
        # Interactive Treemap mapping Region -> Tier -> State
        fig_tree = px.treemap(
            df, 
            path=['Region', 'Market_Tier_Label', 'State'], 
            values='Services_Revenue',
            color='Services_Revenue',
            color_continuous_scale='Blues',
            title="Hierarchical Revenue Density Map ($B)"
        )
        fig_tree.update_layout(template="plotly_dark", plot_bgcolor="#1A1F2B", paper_bgcolor="#1A1F2B", margin=dict(t=30, l=10, r=10, b=10))
        st.plotly_chart(fig_tree, use_container_width=True)

    st.divider()
    
    # 3. Geo-Based Forecasting (Growth Hotspots)
    with st.container(border=True):
        st.markdown("### 3. Predictive Growth Hotspots")
        st.write("We injected a robust multi-feature `RandomForestRegressor` baseline to calculate exact mathematical residuals. States plotting positively here have highly optimized hardware supply chains but are actively underperforming in Software Revenue relative to the algorithmic expectations—earmarking them as **prime targets for hyper-growth.**")
        
        # Train proxy model on current Df natively
        features = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
        pipeline = get_production_pipeline('random_forest', degree=1)
        
        X = df[features]
        y = df['Services_Revenue']
        pipeline.fit(X, y)
        
        df['Predicted_Services'] = pipeline.predict(X)
        df['Growth_Potential_Index'] = df['Predicted_Services'] - df['Services_Revenue']
        
        # Filter for explicit hotspots (Where Model predicts higher than actual)
        hotspots = df[df['Growth_Potential_Index'] > 0].sort_values('Growth_Potential_Index', ascending=False).head(10)
        
        if hotspots.empty:
            st.success("All regions are currently mathematically exceeding or perfectly matching the baseline Random Forest predictive targets.")
        else:
            fig_bar = px.bar(
                hotspots, x='State', y='Growth_Potential_Index', 
                color='Growth_Potential_Index', color_continuous_scale='Teal',
                title="Top 10 Global Expansion Targets (Algorithmic Priority)",
                labels={'Growth_Potential_Index': 'Untapped Software Revenue Potential ($B)'}
            )
            fig_bar.update_layout(template="plotly_dark", plot_bgcolor="#1A1F2B", paper_bgcolor="#1A1F2B", margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_bar, use_container_width=True)
