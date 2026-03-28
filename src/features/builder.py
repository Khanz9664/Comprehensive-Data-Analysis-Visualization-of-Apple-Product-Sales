import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineers new features from existing sales data."""
    if df is None:
        return None
        
    df = df.copy()
    df['Total_Product_Sales'] = df['iPhone_Sales'] + df['iPad_Sales'] + df['Mac_Sales'] + df['Wearables_Sales']
    df['Revenue_Per_Unit'] = df['Services_Revenue'] / df['Total_Product_Sales']
    df['iPhone_Market_Share'] = (df['iPhone_Sales'] / df['Total_Product_Sales']) * 100
    
    return df
