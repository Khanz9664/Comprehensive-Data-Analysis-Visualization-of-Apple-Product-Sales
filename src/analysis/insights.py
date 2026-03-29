import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_business_recommendations(df: pd.DataFrame) -> list:
    """Builds quantitative IF-THEN strategic recommendations actively tracking exact monetary uplifts."""
    recommendations = []
    
    if df.empty:
        return recommendations
        
    products = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
    avail_p = [p for p in products if p in df.columns]
    
    # 1. Hardware Allocation IF-THEN
    if avail_p and 'Services_Revenue' in df.columns:
        revenue_sums = df[avail_p].sum()
        top_p = revenue_sums.idxmax()
        worst_p = revenue_sums.idxmin()
        t_name = top_p.replace('_Sales', '')
        w_name = worst_p.replace('_Sales', '')
        
        try:
            X = df[[worst_p]]
            y = df['Services_Revenue']
            lr = LinearRegression().fit(X, y)
            elasticity = lr.coef_[0]
            
            w_units = df[worst_p].sum()
            t_units = df[top_p].sum()
            share = (w_units / (w_units + t_units)) * 100
            
            projected_unit_increase = w_units * 0.15
            projected_uplift = projected_unit_increase * elasticity * 1000 # To Millions
            
            if share < 20 and projected_uplift > 0:
                recommendations.append({
                    'type': 'warning',
                    'msg': f"**IF** `{w_name}` unit share remains severely saturated (<{share:.1f}%),\n\n**THEN Action:** Reallocate 15% of regional `{t_name}` marketing budget directly into `{w_name}` peripheral bundling.\n\n**💵 Expected Impact:** Resolving this structural deficit mathematically projects an exact **+${projected_uplift:,.1f} Million** recurring uplift in trailing Software Services."
                })
        except:
            pass

    # 2. Regional Expansion Routing
    if 'Region' in df.columns and 'Services_Revenue' in df.columns and 'Total_Product_Sales' in df.columns:
        region_metrics = df.groupby('Region').agg({'Total_Product_Sales': 'sum', 'Services_Revenue': 'sum'})
        region_metrics['Efficiency'] = region_metrics['Services_Revenue'] / region_metrics['Total_Product_Sales']
        
        if len(region_metrics) >= 2:
            highest_eff = region_metrics['Efficiency'].idxmax()
            lowest_eff = region_metrics['Efficiency'].idxmin()
            
            # Calculate mathematical uplift if lowest efficiency region was scaled natively to global baseline average
            global_eff = df['Services_Revenue'].sum() / df['Total_Product_Sales'].sum()
            current_low_rev = region_metrics.loc[lowest_eff, 'Services_Revenue']
            target_low_rev = region_metrics.loc[lowest_eff, 'Total_Product_Sales'] * global_eff
            impact_gap = (target_low_rev - current_low_rev) * 1000 # in Millions
            
            if impact_gap > 0:
                recommendations.append({
                    'type': 'warning',
                    'msg': f"**IF** `{lowest_eff}` continues operating at inherently bottom-tier software monetization efficiency,\n\n**THEN Action:** Deploy `{highest_eff}`'s exact pricing architecture into `{lowest_eff}` immediately executing localized *Apple One* subscription discounts securely at hardware activation boundaries.\n\n**💵 Expected Impact:** Normalizing `{lowest_eff}` strictly to the global baseline efficiency layout mathematically unlocks **+${impact_gap:,.1f} Million** in untapped regional software retention."
                })
            
    # 3. Product Mix Optimization (Wearables -> iPhone attachment rate logic)
    if 'iPhone_Sales' in df.columns and 'Wearables_Sales' in df.columns and 'Services_Revenue' in df.columns:
        attach_rate = df['Wearables_Sales'].sum() / df['iPhone_Sales'].sum()
        if attach_rate < 0.3:
            # Impact of autonomously scaling attach rate sequentially to 40% minimum threshold
            target_wearables = df['iPhone_Sales'].sum() * 0.4
            wearable_deficit = target_wearables - df['Wearables_Sales'].sum()
            
            # Univariate regression safely mapping explicitly targeted sequential elasticity bounds
            try:
                lr2 = LinearRegression().fit(df[['Wearables_Sales']], df['Services_Revenue'])
                w_elasticity = lr2.coef_[0]
                uplift = wearable_deficit * w_elasticity * 1000
                
                if uplift > 0:
                    recommendations.append({
                        'type': 'success',
                        'msg': f"**IF** baseline Wearables attachment rate logically scales to 40% (Currently degraded at `{attach_rate*100:.1f}%`),\n\n**THEN Action:** Force algorithmic hardware lock-in executing *0% point-of-sale financing* specifically on Wearables paired dynamically with new iPhone activations globally.\n\n**💵 Expected Impact:** Bridging the attachment deficit limits mathematically targets a **+${uplift:,.1f} Million** tracking Software Services network elasticity effect."
                    })
            except:
                pass

    return recommendations
