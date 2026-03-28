import streamlit as st
import pandas as pd
from src.visualization.charts import plot_us_map, plot_top_markets

def render(df: pd.DataFrame):
    col1, col2 = st.columns([3, 1])
    
    with col1:
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
        
        map_metric = st.selectbox(
            "Select Mapping Metric",
            ["iPhone_Sales", "Services_Revenue", "Total_Product_Sales", "Revenue_Per_Unit"],
            format_func=lambda x: x.replace("_", " ").title()
        )
        
        fig_map = plot_us_map(df, map_metric, us_state_to_abbrev)
        if fig_map:
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geospatial data matches filter criteria.")
            
    with col2:
        fig_top = plot_top_markets(df, metric=map_metric)
        st.plotly_chart(fig_top, use_container_width=True)
