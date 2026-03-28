import pandas as pd

def generate_business_recommendations(df: pd.DataFrame) -> list:
    """Analyzes dataframe dynamically to yield actionable business insight statements."""
    recommendations = []
    
    if df.empty:
        return recommendations
        
    # Product Dominance Analysis
    products = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
    # Filter available products from df in case some are dropped
    available_products = [p for p in products if p in df.columns]
    
    if available_products:
        revenue_by_product = df[available_products].sum()
        top_product = revenue_by_product.idxmax()
        worst_product = revenue_by_product.idxmin()
        
        top_name = top_product.replace('_Sales', '')
        worst_name = worst_product.replace('_Sales', '')
        
        total_sales = revenue_by_product.sum()
        if total_sales > 0:
            top_share = (revenue_by_product[top_product] / total_sales) * 100
            worst_share = (revenue_by_product[worst_product] / total_sales) * 100
            
            if top_share > 35:
                recommendations.append({
                    'type': 'success',
                    'msg': f"**{top_name} Dominance:** {top_name} leads with {top_share:.1f}% unit share. **Action:** Maintain aggressive flagship marketing spend to protect structural market share."
                })
                
            if worst_share < 15:
                recommendations.append({
                    'type': 'warning',
                    'msg': f"**{worst_name} Underperformance:** {worst_name} yields only {worst_share:.1f}% unit share. **Action:** Investigate pricing friction or bundle {worst_name} hardware with high-tier {top_name} purchases."
                })
                
    # Regional Performance Analysis
    if 'Region' in df.columns and 'Total_Product_Sales' in df.columns:
        region_sales = df.groupby('Region')['Total_Product_Sales'].sum()
        top_region = region_sales.idxmax()
        worst_region = region_sales.idxmin()
        
        recommendations.append({
            'type': 'success',
            'msg': f"**{top_region} Expansion:** {top_region} is the highest performing gross volume region. **Action:** Leverage successful conversion campaigns from this region into emerging neighboring markets."
        })
        
        recommendations.append({
            'type': 'warning',
            'msg': f"**{worst_region} Penetration:** {worst_region} shows the lowest total sales volume. **Action:** Allocate immediate quarterly budget for localized market research to identify cultural or pricing bottlenecks."
        })
        
    # Efficiency Metrics
    if 'Region' in df.columns and 'Revenue_Per_Unit' in df.columns:
        top_efficiency_region = df.groupby('Region')['Revenue_Per_Unit'].mean().idxmax()
        recommendations.append({
            'type': 'success',
            'msg': f"**Premium Pricing Power:** {top_efficiency_region} yields the highest Services Revenue per product unit. **Action:** Analyze consumer purchasing power strategies here for broader international rollout."
        })
        
    return recommendations
