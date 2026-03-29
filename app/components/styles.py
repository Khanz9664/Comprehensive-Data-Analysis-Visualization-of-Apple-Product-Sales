import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global CSS reset applying native SaaS font boundaries */
        html, body, [class*="css"], .stMarkdown, .stText, h1, h2, h3, h4, h5, p, div {
            font-family: 'Inter', sans-serif !important;
        }

        /* ---------------------------------------------------
         * Typography Overrides
         * --------------------------------------------------- */
        h1, h2, h3 {
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
            color: #E5E7EB !important;
        }

        /* ---------------------------------------------------
         * Hide Default UI Junk 
         * --------------------------------------------------- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Adjust global padding */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            max-width: 95% !important;
        }

        /* ---------------------------------------------------
         * Native Modular "Insight Cards" (.stMetric & stVerticalBlockBorderWrapper)
         * --------------------------------------------------- */
        /* Metrics specifically */
        div[data-testid="stMetric"] {
            background-color: #1A1F2B;
            border: 1px solid rgba(255,255,255,0.08); /* Soft dark border */
            border-radius: 12px;
            padding: 20px !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: all 0.2s ease-in-out;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.5);
            border: 1px solid #6366F1; /* Primary hover glow */
        }
        
        /* Native metric labels */
        div[data-testid="stMetricLabel"] {
            font-size: 0.95rem;
            font-weight: 500;
            color: #9CA3AF;
            margin-bottom: 4px;
        }
        
        /* Native metric values */
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem;
            font-weight: 700;
            color: #E5E7EB;
        }
        
        /* Success / Fail metrics mapped */
        div[data-testid="stMetricDelta"] svg {
            /* We trust Streamlit's native pathing, but modify container context bounds if needed */
        }

        /* ---------------------------------------------------
         * st.container() Cards Override
         * --------------------------------------------------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #1A1F2B !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 12px !important;
            padding: 1rem !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
        }
        
        /* ---------------------------------------------------
         * Generic Button Overrides
         * --------------------------------------------------- */
        button[kind="primary"] {
            background-color: #6366F1 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            border: None !important;
            font-weight: 600 !important;
            box-shadow: 0 2px 4px rgba(99, 102, 241, 0.4) !important;
            transition: all 0.2s ease-in-out !important;
        }
        button[kind="primary"]:hover {
            background-color: #4F46E5 !important; /* Deeper Indigo */
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 8px rgba(99, 102, 241, 0.6) !important;
        }
        
        button[kind="secondary"] {
            background-color: rgba(255,255,255,0.05) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: #E5E7EB !important;
            transition: all 0.2s ease-in-out !important;
        }
        button[kind="secondary"]:hover {
            background-color: rgba(255,255,255,0.1) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
        }

        /* ---------------------------------------------------
         * Input Fields (Selectboxes, text inputs)
         * --------------------------------------------------- */
        .stSelectbox div[data-baseweb="select"] > div {
            background-color: rgba(255, 255, 255, 0.05); /* Slight elevation */
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Alerts Overrides to enforce primary/danger/success palettes correctly */
        div[data-testid="stAlert"] {
            border-radius: 8px !important;
            border: None !important;
        }
        </style>
    """, unsafe_allow_html=True)
