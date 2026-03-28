import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_clustering(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Applies KMeans clustering and returns dataframe with cluster assignments."""
    features_for_clustering = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales', 'Services_Revenue']
    X_cluster = df[features_for_clustering].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    # Calculate silhouette score
    silhouette = silhouette_score(X_scaled, cluster_labels)
    
    # Map clusters to descriptive names
    cluster_names = {
        0: '🏆 Premium Markets', 1: '📈 Growing Markets', 2: '⚖️ Balanced Markets', 
        3: '🎯 Niche Markets', 4: '💼 Enterprise Markets', 5: '🌟 Emerging Markets',
        6: '🔥 High-Value Markets', 7: '🌱 Developing Markets'
    }
    df_clustered['Cluster_Name'] = df_clustered['Cluster'].map(cluster_names)
    
    # Create cluster summary
    cluster_summary = df_clustered.groupby('Cluster_Name')[features_for_clustering].mean().round(2)
    
    return df_clustered, cluster_summary, silhouette
