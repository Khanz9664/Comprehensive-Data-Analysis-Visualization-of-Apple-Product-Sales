import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway, pearsonr

def perform_t_test(df: pd.DataFrame, region1: str, region2: str, metric: str):
    data1 = df[df['Region'] == region1][metric]
    data2 = df[df['Region'] == region2][metric]
    
    if len(data1) == 0 or len(data2) == 0:
        return None, None, None

    t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
    
    # Calculate Cohen's d
    effect_size = (data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
    return t_stat, p_value, effect_size

def perform_anova(df: pd.DataFrame, metric: str):
    region_groups = [df[df['Region'] == region][metric] for region in df['Region'].unique()]
    region_groups = [group for group in region_groups if len(group) > 0]
    
    if len(region_groups) >= 2:
        f_stat, p_value = f_oneway(*region_groups)
        return f_stat, p_value
    return None, None

def calculate_correlation(df: pd.DataFrame, col_x: str, col_y: str):
    correlation, p_value = pearsonr(df[col_x], df[col_y])
    return correlation, p_value
