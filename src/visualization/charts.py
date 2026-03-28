import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Core SaaS Palette
SAAS_COLORS = ["#00B4D8", "#0077B6", "#03045E", "#90E0EF", "#CAF0F8"]
DARK_TEMPLATE = "plotly_dark"

def apply_minimal_styling(fig):
    fig.update_layout(
        template=DARK_TEMPLATE,
        font=dict(family="Inter", color="#E0E0E0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, r=20, b=40, l=20)
    )
    return fig

def plot_correlation_matrix(df: pd.DataFrame, columns: list, title="Cross-Product Correlation Matrix"):
    corr_data = df[columns].corr()
    fig = px.imshow(
        corr_data, text_auto=True, aspect="auto", 
        color_continuous_scale=["#03045E", "#00B4D8"],
        title=title
    )
    return apply_minimal_styling(fig)

def plot_radar_chart(df: pd.DataFrame, group_col='Region', metrics=None):
    if metrics is None:
        metrics = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales', 'Services_Revenue']
        
    summary = df.groupby(group_col).agg({col: 'mean' for col in metrics}).reset_index()
    fig = go.Figure()
    
    for i, row in summary.iterrows():
        color = SAAS_COLORS[i % len(SAAS_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=[m.replace('_', ' ').replace('Sales', '').strip() for m in metrics],
            fill='toself', name=row[group_col], opacity=0.7,
            line=dict(color=color)
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, summary.iloc[:, 1:].max().max()]),
            bgcolor="rgba(0,0,0,0)"
        ),
        title=f"{group_col} Performance Comparison",
    )
    return apply_minimal_styling(fig)

def plot_us_map(df: pd.DataFrame, metric: str, state_map: dict):
    usa_df = df[df['Region'] == 'North America'].copy()
    if usa_df.empty: return None
    
    usa_df['StateAbbr'] = usa_df['State'].map(state_map)
    state_data = usa_df.groupby('StateAbbr')[metric].sum().reset_index()
    
    fig = px.choropleth(
        state_data, locations='StateAbbr', locationmode='USA-states', color=metric,
        scope='usa', color_continuous_scale=["#03045E", "#00B4D8"],
        title=f'{metric.replace("_", " ").title()} Distribution (US)',
    )
    fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='rgba(0,0,0,0)'))
    return apply_minimal_styling(fig)

def plot_top_markets(df: pd.DataFrame, metric: str, top_n: int = 10):
    top_markets = df.nlargest(top_n, metric)[['State', 'Region', metric]]
    fig = px.bar(
        top_markets, x=metric, y='State', color='Region', orientation='h',
        color_discrete_sequence=SAAS_COLORS,
        title=f"Top {top_n} Markets by {metric.replace('_', ' ')}"
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    return apply_minimal_styling(fig)

def plot_scatter_with_trend(df: pd.DataFrame, x_col: str, y_col: str, size_col: str = None, color_col='Region'):
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col, size=size_col,
        hover_data=['State'], trendline="ols",
        color_discrete_sequence=SAAS_COLORS,
        title=f"{y_col.replace('_', ' ').title()} vs {x_col.replace('_', ' ').title()}"
    )
    return apply_minimal_styling(fig)

def plot_distribution(df: pd.DataFrame, metric: str, color_col='Region'):
    fig = px.histogram(
        df, x=metric, color=color_col, marginal="box",
        color_discrete_sequence=SAAS_COLORS,
        title=f"Distribution Profile: {metric.replace('_', ' ').title()}"
    )
    return apply_minimal_styling(fig)

def plot_3d_clusters(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, size_col: str):
    fig = px.scatter_3d(
        df, x=x_col, y=y_col, z=z_col, color='Cluster_Name', size=size_col,
        hover_data=['State', 'Region'], 
        color_discrete_sequence=SAAS_COLORS,
        title="3D Market Segmentation"
    )
    return apply_minimal_styling(fig)

def plot_feature_importance(features: list, coefficients: list):
    importance_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients}).sort_values('Coefficient')
    fig = px.bar(
        importance_df, x='Coefficient', y='Feature', orientation='h',
        title='Feature Importance Drivers',
        color='Coefficient', color_continuous_scale=["#03045E", "#00B4D8"]
    )
    return apply_minimal_styling(fig)

def plot_residuals(y_pred, residuals):
    fig = px.scatter(
        x=y_pred, y=residuals, title="Residuals Analysis",
        labels={'x': 'Predicted Values', 'y': 'Residuals'}, trendline="ols",
        color_discrete_sequence=["#00B4D8"]
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#FF4B4B")
    return apply_minimal_styling(fig)

def plot_boxplot(df: pd.DataFrame, metric: str):
    fig = px.box(
        df, y=metric, title=f"Outlier Topology: {metric.replace('_', ' ').title()}",
        color_discrete_sequence=["#00B4D8"]
    )
    return apply_minimal_styling(fig)

def plot_actual_vs_predicted(y_true, y_pred):
    fig = px.scatter(
        x=y_true, y=y_pred, 
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title="Actual vs Predicted Targeting",
        color_discrete_sequence=["#00B4D8"]
    )
    # Target 1:1 Identity Line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="#FF4B4B", dash="dash"))
    return apply_minimal_styling(fig)
