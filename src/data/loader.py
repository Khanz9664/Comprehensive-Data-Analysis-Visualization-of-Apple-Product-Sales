import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Loads data from a CSV and performs initial cleaning."""
    try:
        df = pd.read_csv(path)
        
        # Standardize column names
        df.columns = [col.strip() for col in df.columns]
        df.columns = [
            "State", "Region", "iPhone_Sales", "iPad_Sales", "Mac_Sales",
            "Wearables_Sales", "Services_Revenue"
        ]
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def get_data_quality(df: pd.DataFrame) -> dict:
    """Calculates data quality metrics on the dataframe."""
    if df is None:
        return {}
        
    return {
        'total_records': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
    }
