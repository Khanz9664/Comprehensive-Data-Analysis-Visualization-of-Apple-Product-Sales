import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
        /* Modern Font Injection */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            padding-top: 1rem;
        }
        
        /* Clean SaaS canvas - hide raw Streamlit default bars */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Premium Metric Cards with smooth Hovers */
        div[data-testid="stMetric"] {
            background-color: #1A1C23;
            border: 1px solid #2D313A;
            padding: 1.2rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        div[data-testid="stMetric"]:hover {
            border-color: #00B4D8;
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,180,216,0.15);
        }
        div[data-testid="stMetricValue"] {
            color: #FFFFFF;
            font-size: 2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        div[data-testid="stMetricLabel"] {
            color: #8B949E;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        div[data-testid="stMetricDelta"] > div {
            font-weight: 600;
        }
        
        /* Header Normalization */
        h1, h2, h3, h4, h5, h6 {
            color: #E6EDF3 !important;
            font-weight: 600 !important;
        }
        
        /* Deep Dark Workspace */
        .stApp {
            background-color: #0D1117;
        }
        
        /* Distinct Flat Sidebar */
        [data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #30363D;
        }
        
        /* Dataframes embedded cleanly */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #30363D;
        }
    </style>
    """, unsafe_allow_html=True)
