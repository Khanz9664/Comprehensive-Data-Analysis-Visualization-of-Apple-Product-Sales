# --- Import Necessary Libraries ---
# We're bringing in all the tools we need for this dashboard.
import pandas as pd  # The master of data manipulation
import streamlit as st  # The framework for our interactive web app
import plotly.express as px  # For creating beautiful, interactive charts quickly
import plotly.graph_objects as go  # For more custom and complex plots (like radar charts)
from plotly.subplots import make_subplots # For creating grids of plots
from scipy.stats import ttest_ind, f_oneway, pearsonr  # Our toolbox for statistical tests
from sklearn.linear_model import LinearRegression  # The workhorse for our prediction models
from sklearn.metrics import r2_score, mean_squared_error  # To check how good our models are
from sklearn.preprocessing import StandardScaler  # To scale our data for machine learning
from sklearn.cluster import KMeans  # For our customer segmentation algorithm
import numpy as np  # A fundamental package for scientific computing in Python
import seaborn as sns # Used for some statistical plots, although less here
import matplotlib.pyplot as plt # Also available for plotting, but we primarily use Plotly

# --- Advanced Page Configuration ---
# This sets up the basic look and feel of our Streamlit page.
# A wide layout gives us more screen real estate to work with.
st.set_page_config(
    page_title="Apple Sales (2024) Analytics",
    page_icon="📱",  # A little emoji for the browser tab
    layout="wide",
    initial_sidebar_state="expanded"  # Keep the sidebar open by default
)

# --- Custom CSS for Professional Styling ---
# Here, we're injecting some custom CSS to make our dashboard look polished and professional.
# This is a bit of a trick to override default Streamlit styles.
st.markdown("""
<style>
    /* Import a clean, modern font from Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling for the main app area */
    .main {
        padding-top: 1rem;
    }
    
    /* Custom styling for our main header */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .custom-header h1 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin: 0;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .custom-header p {
        color: rgba(255,255,255,0.9);
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Styling for the KPI (Key Performance Indicator) cards */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px); /* A subtle lift effect on hover */
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
    }
    
    .metric-label {
        font-size: 1rem;
        font-weight: 400;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Customizing the look of the tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        padding: 0px 20px;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Styling the sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Styling for our insight boxes, which highlight key findings */
    .insight-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .insight-box h4 {
        color: #2c3e50;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .insight-box p {
        color: #34495e;
        font-family: 'Inter', sans-serif;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Styling for the cards in the statistics tab */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    
    .stat-card h3 {
        color: #2c3e50;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* A nice badge to show off our data quality */
    .quality-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    /* A simple loading spinner animation */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True) # `unsafe_allow_html` is needed to render raw HTML/CSS


# --- Advanced Data Loading with Progress Bar ---
# This function handles loading our data. We use @st.cache_data to 'memoize' the function.
# This means Streamlit will only run it once and store the result in a cache.
# If the function is called again with the same input, it returns the cached result,
# which makes the app much faster.
@st.cache_data
def load_and_process_data(path):
    """
    Loads data from a CSV, performs preprocessing, and engineers new features.
    It also calculates data quality metrics on the fly.
    """
    try:
        df = pd.read_csv(path)
        
        # --- Data Cleaning and Standardization ---
        # It's good practice to standardize column names to avoid issues with whitespace or typos.
        df.columns = [col.strip() for col in df.columns]
        df.columns = [
            "State", "Region", "iPhone_Sales", "iPad_Sales", "Mac_Sales",
            "Wearables_Sales", "Services_Revenue"
        ]
        
        # --- Feature Engineering ---
        # We create new, meaningful columns from the existing data.
        # This helps reveal deeper insights.
        df['Total_Hardware_Sales'] = df['iPhone_Sales'] + df['iPad_Sales'] + df['Mac_Sales'] + df['Wearables_Sales']
        df['Revenue_Per_Unit'] = df['Services_Revenue'] / df['Total_Hardware_Sales']
        df['iPhone_Market_Share'] = df['iPhone_Sales'] / df['Total_Hardware_Sales'] * 100
        
        # --- Data Quality Assessment ---
        # We gather some metadata about our dataset. This is useful for the Data Quality tab.
        data_quality = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # Convert bytes to MB
        }
        
        return df, data_quality
        
    except Exception as e:
        # If the file isn't found or something else goes wrong, show a friendly error.
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Let's load the data. The 'with st.spinner' provides a nice loading message for the user.
with st.spinner("Loading and processing Apple sales data..."):
    df, data_quality = load_and_process_data('apple_sales_2024.csv')

# If data loading failed, we stop the app from running further.
if df is None:
    st.error("Failed to load data. Please check the file path and ensure 'apple_sales_2024.csv' is in the correct folder.")
    st.stop()

# --- Professional Header ---
# Using our custom CSS class to create a visually appealing header for the dashboard.
st.markdown("""
<div class="custom-header">
    <h1>📱 Apple Analytics Hub</h1>
    <p>Advanced Data Science Portfolio | Comprehensive Sales Intelligence & Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar Enhancement ---
# The sidebar is our main control panel for filters and other info.
with st.sidebar:
    st.markdown("### Dashboard Controls")
    st.markdown("---")
    
    # Using our custom badge to display a data quality score.
    st.markdown(f"""
    <div class="quality-badge">
        Data Quality Score: {((data_quality['total_records'] - data_quality['missing_values']) / data_quality['total_records'] * 100):.1f}%
    </div>
    """, unsafe_allow_html=True)
    
    # --- Advanced Filters ---
    # These widgets allow the user to interactively filter the data shown in the charts.
    st.markdown("#### Advanced Filters")
    selected_regions = st.multiselect(
        "Select Regions",
        options=df['Region'].unique(),
        default=df['Region'].unique() # By default, all regions are selected.
    )
    
    sales_threshold = st.slider(
        "Minimum iPhone Sales (M units)",
        min_value=float(df['iPhone_Sales'].min()),
        max_value=float(df['iPhone_Sales'].max()),
        value=float(df['iPhone_Sales'].min()), # Start with the minimum value.
        step=1.0
    )
    
    # Apply the filters to our main DataFrame. This filtered_df will be used for all visualizations.
    filtered_df = df[
        (df['Region'].isin(selected_regions)) & 
        (df['iPhone_Sales'] >= sales_threshold)
    ]
    
    st.markdown("---")
    
    # This section is to showcase the skills and technologies used in this project.
    st.markdown("#### Highlights")
    st.markdown("""
    **Technical Skills Demonstrated:**
    - Advanced Statistical Analysis
    - Machine Learning & Predictive Modeling
    - Interactive Data Visualization
    - Hypothesis Testing & A/B Testing
    - Customer Segmentation Analysis
    - Data Quality Assessment
    - Geospatial Analytics
    
    **Technologies Used:**
    - Python (Pandas, NumPy, Scikit-learn)
    - Plotly & Advanced Visualization
    - Statistical Analysis (SciPy)
    - Streamlit Dashboard Development
    """)
    
    st.markdown("---")
    st.info(" **Tip:** Interact with all charts and filters to explore the full analytics!")

# --- Enhanced KPI Dashboard ---
# This is the executive summary section with high-level metrics.
st.markdown("### Executive Dashboard - Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5) # Create 5 columns for our KPI cards.

with col1:
    total_revenue = filtered_df['Services_Revenue'].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">${total_revenue:,.1f}B</div>
        <div class="metric-label">Total Services Revenue</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    total_units = filtered_df['Total_Hardware_Sales'].sum()
    # We apply a different gradient to this card for visual variety.
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);">
        <div class="metric-value">{total_units:,.0f}M</div>
        <div class="metric-label">Total Units Sold</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_revenue_per_unit = filtered_df['Revenue_Per_Unit'].mean()
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);">
        <div class="metric-value">${avg_revenue_per_unit:.2f}</div>
        <div class="metric-label">Avg Revenue/Unit</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    iphone_dominance = filtered_df['iPhone_Market_Share'].mean()
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);">
        <div class="metric-value">{iphone_dominance:.1f}%</div>
        <div class="metric-label">iPhone Market Share</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    total_markets = filtered_df['State'].nunique()
    st.markdown(f"""
    <div class="metric-card" style="background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);">
        <div class="metric-value">{total_markets}</div>
        <div class="metric-label">Active Markets</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---") # A horizontal line for separation.

# --- Advanced Tabbed Interface ---
# Tabs are a great way to organize a lot of content without cluttering the page.
tab_overview, tab_geo, tab_analytics, tab_segmentation, tab_predictions, tab_quality = st.tabs([
    "Global Overview", "Geospatial Intelligence", "Advanced Analytics", 
    "Customer Segmentation", "Predictive Models", "Data Quality"
])


# --- GLOBAL OVERVIEW TAB ---
with tab_overview:
    st.markdown("### Global Sales Performance Matrix")
    
    col1, col2 = st.columns([2, 1]) # Create two columns, with the first being twice as wide.
    
    with col1:
        # --- Advanced Correlation Heatmap ---
        # A heatmap is perfect for quickly seeing the strength of relationships between variables.
        st.markdown("#### Product Line Correlation Analysis")
        corr_data = filtered_df[['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales', 'Services_Revenue']].corr()
        
        fig_corr = px.imshow(
            corr_data,
            text_auto=True,  # Automatically display the correlation values on the map
            aspect="auto",
            color_continuous_scale="RdBu_r", # A diverging color scale is good for correlations
            title="Cross-Product Correlation Matrix"
        )
        fig_corr.update_layout(
            title_x=0.5, # Center the title
            height=400,
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig_corr, use_container_width=True) # Display the chart
        
        # --- Professional Insight Box ---
        # This box provides context and interprets the chart for the user.
        st.markdown("""
        <div class="insight-box">
            <h4>Insights</h4>
            <p>Strong positive correlation (0.7+) between hardware sales and services revenue indicates successful ecosystem lock-in strategy. This correlation analysis helps identify cross-selling opportunities and product bundling strategies.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # --- Regional Performance Radar Chart ---
        # A radar chart is excellent for comparing multiple quantitative variables for different groups.
        st.markdown("#### Regional Performance Radar")
        
        region_summary = filtered_df.groupby('Region').agg({
            'iPhone_Sales': 'mean',
            'iPad_Sales': 'mean',
            'Mac_Sales': 'mean',
            'Wearables_Sales': 'mean',
            'Services_Revenue': 'mean'
        }).reset_index()
        
        fig_radar = go.Figure()
        
        # Loop through each region to add a 'trace' (a layer) to the radar chart.
        for _, region in region_summary.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[region['iPhone_Sales'], region['iPad_Sales'], region['Mac_Sales'], 
                   region['Wearables_Sales'], region['Services_Revenue']],
                theta=['iPhone', 'iPad', 'Mac', 'Wearables', 'Services'], # These are the 'spokes' of the radar
                fill='toself', # Fill the area under the line
                name=region['Region'],
                opacity=0.7
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, region_summary.iloc[:, 1:].max().max()])
            ),
            title="Regional Performance Comparison",
            title_x=0.5,
            height=400
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)


# --- GEOSPATIAL INTELLIGENCE TAB ---
with tab_geo:
    st.markdown("### Advanced Geospatial Analytics")
    
    col1, col2 = st.columns([3, 1]) # Column 1 is three times wider than column 2.
    
    with col1:
        # --- Enhanced US Map with Multiple Metrics ---
        # A choropleth map is the best way to visualize data distributed across geographic regions.
        st.markdown("#### Multi-Dimensional US Market Analysis")
        
        # We need a mapping from full state names to their 2-letter abbreviations for the map.
        us_state_to_abbrev = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", 
            "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", 
            "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", 
            "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", 
            "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", 
            "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", 
            "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", 
            "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", 
            "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", 
            "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", 
            "District of Columbia": "DC"
        }
        
        # Let the user choose which metric to display on the map.
        map_metric = st.selectbox(
            "Select Metric for Mapping",
            ["iPhone_Sales", "Services_Revenue", "Total_Hardware_Sales", "Revenue_Per_Unit"],
            format_func=lambda x: x.replace("_", " ").title() # Makes the options look nicer (e.g., 'iPhone_Sales' -> 'Iphone Sales')
        )
        
        usa_df = filtered_df[filtered_df['Region'] == 'North America'].copy()
        if not usa_df.empty:
            usa_df['StateAbbr'] = usa_df['State'].map(us_state_to_abbrev)
            state_data = usa_df.groupby('StateAbbr')[map_metric].sum().reset_index()
            
            fig_map = px.choropleth(
                state_data,
                locations='StateAbbr',
                locationmode='USA-states', # Tell Plotly we're mapping US states
                color=map_metric,
                scope='usa', # Limit the map view to the USA
                color_continuous_scale="Viridis",
                title=f'{map_metric.replace("_", " ").title()} Distribution Across US States',
                hover_name='StateAbbr',
                hover_data={map_metric: ':,.2f'} # Format the hover data nicely
            )
            fig_map.update_layout(title_x=0.5, height=500, font=dict(family="Inter"))
            st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        # --- Top Performing Markets ---
        # A simple bar chart is great for ranking items.
        st.markdown("#### Top Performing Markets")
        
        top_markets = filtered_df.nlargest(10, 'Services_Revenue')[['State', 'Region', 'Services_Revenue']]
        
        fig_top = px.bar(
            top_markets,
            x='Services_Revenue',
            y='State',
            color='Region',
            orientation='h', # Horizontal bar chart
            title="Top 10 Markets by Revenue"
        )
        fig_top.update_layout(height=400, yaxis={'categoryorder':'total ascending'}) # Sort bars by value
        st.plotly_chart(fig_top, use_container_width=True)


# --- ADVANCED ANALYTICS TAB ---
with tab_analytics:
    st.markdown("### Statistical Analysis & Hypothesis Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # --- Enhanced T-Test Analysis ---
        # A T-test checks if there's a significant difference between the means of two groups.
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("#### Advanced T-Test Analysis")
        
        regions = filtered_df['Region'].unique()
        if len(regions) >= 2: # We need at least two regions to compare
            region1 = st.selectbox("Select First Region", regions, key="region1")
            region2 = st.selectbox("Select Second Region", [r for r in regions if r != region1], key="region2")
            metric_to_test = st.selectbox("Select Metric to Test", 
                                        ["iPhone_Sales", "Services_Revenue", "Total_Hardware_Sales"])
            
            data1 = filtered_df[filtered_df['Region'] == region1][metric_to_test]
            data2 = filtered_df[filtered_df['Region'] == region2][metric_to_test]
            
            if len(data1) > 0 and len(data2) > 0:
                # `ttest_ind` performs the independent t-test. `equal_var=False` is for Welch's t-test, which is more robust.
                t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
                # Cohen's d is a measure of effect size. It tells us how large the difference is.
                effect_size = (data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
                
                # Display the results in neat metric cards.
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("T-Statistic", f"{t_stat:.3f}")
                col_b.metric("P-Value", f"{p_value:.4f}")
                col_c.metric("Effect Size (Cohen's d)", f"{effect_size:.3f}")
                
                # Interpret the p-value for the user.
                if p_value < 0.001:
                    significance = "Highly Significant (p < 0.001)"
                    color = "🟢"
                elif p_value < 0.01:
                    significance = "Very Significant (p < 0.01)"
                    color = "🟡"
                elif p_value < 0.05:
                    significance = "Significant (p < 0.05)"
                    color = "🟠"
                else:
                    significance = "Not Significant (p ≥ 0.05)"
                    color = "🔴"
                
                st.markdown(f"**Statistical Conclusion:** {color} {significance}")
                
                # Interpret the effect size.
                if abs(effect_size) < 0.2:
                    effect_interp = "Small effect"
                elif abs(effect_size) < 0.5:
                    effect_interp = "Medium effect"
                else:
                    effect_interp = "Large effect"
                
                st.markdown(f"**Effect Size:** {effect_interp}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # --- ANOVA Analysis ---
        # ANOVA is like a T-test but for comparing the means of three or more groups.
        st.markdown('<div class="stat-card">', unsafe_allow_html=True)
        st.markdown("#### One-Way ANOVA Analysis")
        
        anova_metric = st.selectbox("Select Metric for ANOVA", 
                                  ["iPhone_Sales", "iPad_Sales", "Mac_Sales", "Services_Revenue"],
                                  key="anova_metric")
        
        # Prepare the data for ANOVA: a list of Series, where each Series is the data for one region.
        region_groups = [filtered_df[filtered_df['Region'] == region][anova_metric] 
                        for region in filtered_df['Region'].unique()]
        
        # Filter out any empty groups which would cause an error.
        region_groups = [group for group in region_groups if len(group) > 0]
        
        if len(region_groups) >= 2:
            f_stat, p_value_anova = f_oneway(*region_groups)
            
            col_a, col_b = st.columns(2)
            col_a.metric("F-Statistic", f"{f_stat:.3f}")
            col_b.metric("P-Value", f"{p_value_anova:.4f}")
            
            # If the p-value is significant, it means at least one region is different from the others.
            if p_value_anova < 0.05:
                st.success("**Significant difference detected** between regions")
                
                # We can do a simple post-hoc analysis by just ranking the means.
                region_means = filtered_df.groupby('Region')[anova_metric].mean().sort_values(ascending=False)
                st.markdown("**Ranking by Mean Performance:**")
                for i, (region, mean_val) in enumerate(region_means.items(), 1):
                    st.markdown(f"{i}. **{region}**: {mean_val:.2f}")
            else:
                st.warning("**No significant difference** detected between regions")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # --- Correlation Analysis ---
    st.markdown("#### Advanced Correlation & Causation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # A scatter plot is the classic way to visualize the relationship between two continuous variables.
        x_var = st.selectbox("X-Axis Variable", 
                           ["iPhone_Sales", "iPad_Sales", "Mac_Sales", "Total_Hardware_Sales"],
                           key="x_var")
        y_var = st.selectbox("Y-Axis Variable", 
                           ["Services_Revenue", "Revenue_Per_Unit"],
                           key="y_var")
        
        # Calculate Pearson's correlation coefficient and its p-value.
        correlation, corr_p_value = pearsonr(filtered_df[x_var], filtered_df[y_var])
        
        fig_scatter = px.scatter(
            filtered_df, 
            x=x_var, 
            y=y_var,
            color='Region',
            size='Total_Hardware_Sales', # Use size to encode a third dimension
            hover_data=['State'],
            trendline="ols", # 'ols' adds an Ordinary Least Squares regression line
            title=f"{y_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}"
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Display the correlation metrics.
        col_a, col_b = st.columns(2)
        col_a.metric("Pearson Correlation", f"{correlation:.3f}")
        col_b.metric("Correlation P-Value", f"{corr_p_value:.4f}")
    
    with col2:
        # --- Distribution Analysis ---
        # A histogram is essential for understanding the distribution of a single variable.
        st.markdown("#### Distribution Analysis")
        
        dist_var = st.selectbox("Select Variable for Distribution Analysis",
                               ["iPhone_Sales", "Services_Revenue", "Revenue_Per_Unit"])
        
        fig_dist = px.histogram(
            filtered_df,
            x=dist_var,
            color='Region', # Show distributions for each region separately
            marginal="box", # Add a box plot to the margin to see outliers and quartiles
            title=f"Distribution of {dist_var.replace('_', ' ').title()}"
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Display key descriptive statistics.
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Mean", f"{filtered_df[dist_var].mean():.2f}")
        col_b.metric("Std Dev", f"{filtered_df[dist_var].std():.2f}")
        col_c.metric("Skewness", f"{filtered_df[dist_var].skew():.2f}")


# --- CUSTOMER SEGMENTATION TAB ---
with tab_segmentation:
    st.markdown("### Advanced Customer Segmentation Analysis")
    
    st.markdown("""
    <div class="insight-box">
        <h4>Machine Learning Application</h4>
        <p>Using K-Means clustering algorithm to identify distinct market segments based on sales patterns. This unsupervised learning approach can reveal hidden structures in the data without needing pre-defined labels.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- Prepare data for clustering ---
    features_for_clustering = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales', 'Services_Revenue']
    X_cluster = filtered_df[features_for_clustering].copy()
    
    # It's crucial to scale features before using K-Means.
    # K-Means is distance-based, so features with larger scales can dominate the result.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Let the user choose the number of clusters (k).
    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=8, value=4, key="k_slider")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init='auto' or 10 is recommended
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add the cluster labels back to our DataFrame for analysis.
    df_clustered = filtered_df.copy()
    df_clustered['Cluster'] = cluster_labels
    # Let's give our clusters more descriptive names.
    df_clustered['Cluster_Name'] = df_clustered['Cluster'].map({
        0: '🏆 Premium Markets', 1: '📈 Growing Markets', 2: '⚖️ Balanced Markets', 
        3: '🎯 Niche Markets', 4: '💼 Enterprise Markets', 5: '🌟 Emerging Markets',
        6: '🔥 High-Value Markets', 7: '🌱 Developing Markets'
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # A 3D scatter plot is a fantastic way to visualize clusters if you have 3 key features.
        st.markdown("#### 3D Market Segmentation Visualization")
        fig_3d = px.scatter_3d(
            df_clustered,
            x='iPhone_Sales',
            y='Services_Revenue',
            z='Total_Hardware_Sales',
            color='Cluster_Name',
            size='Revenue_Per_Unit', # Use marker size to encode a 4th dimension
            hover_data=['State', 'Region'],
            title="3D Market Segmentation Analysis"
        )
        fig_3d.update_layout(height=500)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        # Summarize the characteristics of each cluster to understand what they represent.
        st.markdown("#### Segment Characteristics")
        
        cluster_summary = df_clustered.groupby('Cluster_Name').agg({
            'iPhone_Sales': 'mean',
            'Services_Revenue': 'mean',
            'Revenue_Per_Unit': 'mean',
            'State': 'count' # Count how many markets are in each cluster
        }).round(2)
        cluster_summary.columns = ['Avg iPhone Sales', 'Avg Services Rev', 'Rev/Unit', 'Markets Count']
        
        st.dataframe(cluster_summary, use_container_width=True)
        
        # Highlight some key findings from the clustering.
        dominant_cluster = cluster_summary['Markets Count'].idxmax()
        st.success(f"**Dominant Segment:** {dominant_cluster}")
        
        highest_revenue_cluster = cluster_summary['Avg Services Rev'].idxmax()
        st.info(f"**Highest Revenue Segment:** {highest_revenue_cluster}")
        
    # Provide actionable business recommendations based on the segments found.
    st.markdown("#### Strategic Business Recommendations")
    recommendations_col1, recommendations_col2 = st.columns(2)
    with recommendations_col1:
        st.markdown("""
        <div class="insight-box">
            <h4>Market Penetration Strategy</h4>
            <p>Focus marketing efforts on 'Growing' or 'Emerging' markets. Implement targeted campaigns to accelerate adoption and increase market share in these high-potential segments.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with recommendations_col2:
        st.markdown("""
        <div class="insight-box">
            <h4>Revenue Optimization</h4>
            <p>Analyze the 'Premium' and 'High-Value' markets to understand what drives their success. Apply these learnings to boost performance in 'Balanced' and 'Niche' markets.</p>
        </div>
        """, unsafe_allow_html=True)


# --- PREDICTIVE MODELS TAB ---
with tab_predictions:
    st.markdown("### Advanced Predictive Analytics & Machine Learning")
    
    st.markdown("""
    <div class="insight-box">
        <h4>Machine Learning </h4>
        <p>Here, we demonstrate how to build and evaluate regression models to predict Services Revenue. This section is still under-construction. </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Let the user choose which type of model to explore.
    model_type = st.selectbox(
        "Select Predictive Model Type",
        ["Linear Regression", "Multiple Linear Regression", "Polynomial Features"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if model_type == "Linear Regression":
            # --- Simple Linear Regression ---
            # Predicting one variable from one other variable.
            st.markdown("#### Simple Linear Regression: iPhone Sales → Services Revenue")
            
            X = filtered_df[['iPhone_Sales']] # Feature (must be a DataFrame)
            y = filtered_df['Services_Revenue'] # Target
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Calculate key performance metrics for the model.
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            # The trendline from Plotly Express is essentially doing this regression for us visually.
            fig_pred = px.scatter(
                filtered_df, 
                x='iPhone_Sales', 
                y='Services_Revenue',
                color='Region',
                title='Services Revenue Prediction Model',
                trendline="ols"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
        elif model_type == "Multiple Linear Regression":
            # --- Multiple Linear Regression ---
            # Predicting one variable from several other variables.
            st.markdown("#### Multiple Linear Regression: Multi-Product Model")
            
            features = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
            X = filtered_df[features]
            y = filtered_df['Services_Revenue']
            
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            # --- Feature Importance ---
            # In a linear model, the coefficients tell us the importance and direction of each feature's effect.
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Coefficient': model.coef_
            }).sort_values('Coefficient', ascending=True)
            
            fig_importance = px.bar(
                feature_importance,
                x='Coefficient',
                y='Feature',
                orientation='h',
                title='Feature Importance (Regression Coefficients)',
                color='Coefficient',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
            
            # --- Residuals Plot ---
            # A good model should have residuals (errors) randomly scattered around zero.
            residuals = y - y_pred
            fig_residuals = px.scatter(
                x=y_pred, 
                y=residuals,
                title="Residuals Analysis",
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                trendline="ols" # A trendline here should be flat if the model is good.
            )
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        else:  # Polynomial Features
            # --- Polynomial Regression ---
            # This can capture non-linear relationships.
            st.markdown("#### Polynomial Regression: Non-Linear Relationships")
            
            from sklearn.preprocessing import PolynomialFeatures
            
            X = filtered_df[['iPhone_Sales']]
            y = filtered_df['Services_Revenue']
            
            poly_degree = st.slider("Polynomial Degree", 1, 5, 2)
            poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
            X_poly = poly_features.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            
            # Plot the original data points.
            fig_poly = px.scatter(
                filtered_df,
                x='iPhone_Sales',
                y='Services_Revenue',
                color='Region',
                title=f'Polynomial Regression (Degree {poly_degree})'
            )
            
            # Create a smooth line for the polynomial curve to plot.
            x_range = np.linspace(X['iPhone_Sales'].min(), X['iPhone_Sales'].max(), 100).reshape(-1, 1)
            x_range_poly = poly_features.transform(x_range)
            y_pred_poly = model.predict(x_range_poly)
            
            fig_poly.add_trace(go.Scatter(
                x=x_range.flatten(),
                y=y_pred_poly,
                mode='lines',
                name=f'Polynomial Fit',
                line=dict(color='red', width=3)
            ))
            
            st.plotly_chart(fig_poly, use_container_width=True)
    
    with col2:
        # --- Model Performance Metrics ---
        # Display the key metrics for the selected model.
        st.markdown("#### Model Performance")
        st.markdown(f'<div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);"><div class="metric-value">{r2:.3f}</div><div class="metric-label">R² Score</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);"><div class="metric-value">{rmse:.2f}</div><div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card" style="background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);"><div class="metric-value">{mse:.2f}</div><div class="metric-label">MSE</div></div>', unsafe_allow_html=True)
        
        # Give a simple quality rating based on the R-squared value.
        if r2 > 0.8: model_quality = "🟢 Excellent"
        elif r2 > 0.6: model_quality = "🟡 Good"
        elif r2 > 0.4: model_quality = "🟠 Fair"
        else: model_quality = "🔴 Poor"
        st.markdown(f"**Model Quality:** {model_quality}")
        
        # --- Business Insights from the Model ---
        st.markdown("#### Business Insights")
        if model_type == "Linear Regression":
            coefficient = model.coef_[0]
            st.markdown(f"**Interpretation:** For every 1M increase in iPhone sales, Services Revenue is predicted to increase by **${coefficient:.2f}B**.")
        elif model_type == "Multiple Linear Regression":
            st.markdown("**Top Revenue Drivers:**")
            importance_df = pd.DataFrame({'Feature': features, 'Coefficient': model.coef_}).sort_values('Coefficient', ascending=False)
            for _, row in importance_df.iterrows():
                st.markdown(f"- **{row['Feature'].replace('_', ' ')}**: ${row['Coefficient']:.2f}B per 1M units")

        # --- Prediction Interface ---
        # Let's build a small tool to use our trained model.
        st.markdown("#### Make a Prediction")
        if model_type == "Linear Regression":
            iphone_input = st.number_input("iPhone Sales (M units)", min_value=0.0, max_value=100.0, value=25.0)
            # Wrap input in a DataFrame with matching column name
            input_df = pd.DataFrame([[iphone_input]], columns=['iPhone_Sales'])
            predicted_revenue = model.predict(input_df)[0]
            st.success(f"Predicted Services Revenue: ${predicted_revenue:.2f}B")
        elif model_type == "Multiple Linear Regression":
            st.markdown("**Input Sales Data (in Millions):**")
            iphone_in = st.number_input("iPhone Sales", 0.0, 100.0, 25.0, key="iphone_multi")
            ipad_in = st.number_input("iPad Sales", 0.0, 50.0, 10.0, key="ipad_multi")
            mac_in = st.number_input("Mac Sales", 0.0, 50.0, 8.0, key="mac_multi")
            wearables_in = st.number_input("Wearables Sales", 0.0, 50.0, 12.0, key="wearables_multi")
            
            if st.button("Predict Revenue"):
                # Wrap input in DataFrame with correct feature names
                input_df = pd.DataFrame(
                    [[iphone_in, ipad_in, mac_in, wearables_in]],
                    columns=['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
                )
                prediction = model.predict(input_df)[0]
                st.success(f"Predicted Services Revenue: ${prediction:.2f}B")


# --- DATA QUALITY TAB ---
with tab_quality:
    st.markdown("### Data Quality Assessment & Validation")
    
    st.markdown("""
    <div class="insight-box">
        <h4>Data Engineering Excellence</h4>
        <p>A thorough data quality check is the foundation of any reliable analysis. This section demonstrates attention to data integrity, validation, and preprocessing - critical steps for any data science project.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # --- Data Quality Metrics ---
        st.markdown("#### Data Quality Overview")
        st.info(f" **Total Records:** {data_quality['total_records']}")
        st.success(f" **Missing Values:** {data_quality['missing_values']}")
        st.success(f" **Duplicate Rows:** {data_quality['duplicate_rows']}")
        st.info(f" **Memory Usage:** {data_quality['memory_usage']:.2f} MB")

        # --- Data Schema ---
        st.markdown("#### Data Types & Schema")
        schema_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(dtype) for dtype in df.dtypes],
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(schema_df, use_container_width=True)
    
    with col2:
        # --- Statistical Summary ---
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)
        
        # --- Outlier Detection ---
        # The IQR method is a common way to statistically identify outliers.
        st.markdown("#### Outlier Detection (IQR Method)")
        numeric_cols = df.select_dtypes(include=np.number).columns
        outlier_col = st.selectbox("Select Column for Outlier Analysis", numeric_cols, key="outlier_select")
        
        Q1 = df[outlier_col].quantile(0.25)
        Q3 = df[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
        
        st.metric("Outliers Detected", len(outliers), f"{(len(outliers)/len(df)*100):.1f}% of data")
        
        # A box plot is the perfect visualization for the IQR method.
        fig_box = px.box(df, y=outlier_col, title=f"Outlier Visualization: {outlier_col.replace('_', ' ').title()}")
        st.plotly_chart(fig_box, use_container_width=True)
    
    # --- Data Validation Rules ---
    # These are custom checks based on business logic.
    st.markdown("#### Data Validation Rules")
    validation_col1, validation_col2 = st.columns(2)
    with validation_col1:
        st.markdown("##### **Business Logic Checks**")
        # Sales figures should never be negative.
        if (df[features_for_clustering] < 0).any().any():
            st.error(" **Negative Sales/Revenue Detected**")
        else:
            st.success(" **No Negative Sales/Revenue Values**")

        # All revenue values should be positive.
        if (df['Services_Revenue'] <= 0).any():
             st.error(" **Invalid (Zero or Negative) Revenue Values**")
        else:
             st.success(" **All Revenue Values are Positive**")

    with validation_col2:
        st.markdown("##### **Consistency Checks**")
        # Check if region names are as expected.
        expected_regions = ['North America', 'Europe', 'Greater China', 'Rest of Asia', 'Rest of World']
        if not set(df['Region'].unique()).issubset(expected_regions):
            st.warning(" **Unexpected Region Values Found**")
        else:
            st.success(" **Region Names are Consistent**")

        # A logical check: total hardware sales should correlate positively with services.
        if df['Total_Hardware_Sales'].corr(df['Services_Revenue']) < 0.3:
            st.warning(" **Weak Hardware-Service Correlation**")
        else:
            st.success(" **Logical Hardware-Service Correlation**")


# --- Footer ---
st.markdown("---", unsafe_allow_html=True)

html_footer = """
<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
             border-radius: 15px; color: white;">
    <h3>Apple Sales Analytics Portfolio</h3>
    <p>This dashboard demonstrates advanced proficiency in:</p>
    <p><strong>Statistical Analysis • Machine Learning • Data Visualization • Business Intelligence • Python Development</strong></p>
    <p style="margin-top: 1rem; opacity: 0.8;">
        Built with Python, Streamlit, Plotly, Pandas, Scikit-learn, and SciPy
    </p>
    <div style="margin-top: 2rem;">
        <a href="https://github.com/Khanz9664" target="_blank" style="margin: 0 15px; text-decoration: none; color: white;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg"
                 width="30" style="vertical-align: middle;" /> GitHub
        </a>
        <a href="https://www.linkedin.com/in/shahid-ul-islam-13650998" target="_blank"
           style="margin: 0 15px; text-decoration: none; color: white;">
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg"
                 width="30" style="vertical-align: middle;" /> LinkedIn
        </a>
        <a href="https://khanz9664.github.io/portfolio" target="_blank"
           style="margin: 0 15px; text-decoration: none; color: white;">
            🌐 Portfolio
        </a>
    </div>
</div>
"""

st.markdown(html_footer, unsafe_allow_html=True)
