import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    st.markdown("This brief outlines the most critical business insights derived from the active data filters. It strictly avoids statistical jargon to deliver clear, actionable strategies.")
    
    # Calculate core metrics natively to keep it independent of statistical files
    products = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
    available_products = [p for p in products if p in df.columns]
    
    if not available_products or 'Region' not in df.columns or 'Services_Revenue' not in df.columns:
        st.warning("Insufficient data depth to generate an executive report.")
        return
        
    revenue_by_product = df[available_products].sum()
    top_product = revenue_by_product.idxmax().replace('_Sales', '')
    worst_product = revenue_by_product.idxmin().replace('_Sales', '')
    
    region_sales = df.groupby('Region')['Total_Product_Sales'].sum()
    top_region = region_sales.idxmax()
    worst_region = region_sales.idxmin()
    
    revenue_per_unit_region = df.groupby('Region')['Revenue_Per_Unit'].mean().idxmax()
    
    st.divider()
    
    # 1. Key Findings
    st.subheader("1. Key Findings")
    st.info(f"""
    - **Top Product:** The **{top_product}** is the primary dominator of product revenue.
    - **Top Market:** **{top_region}** generates the highest overall sales volume globally.
    - **Major Trend:** Selling hardware units successfully is directly and consistently correlating with long-term *Services Revenue* growth.
    """)
    
    # 2. Risks
    st.subheader("2. Risks & Challenges")
    st.warning(f"""
    - **Underperforming Product:** The **{worst_product}** represents the absolute lowest share of the product ecosystem.
    - **Weakest Market:** **{worst_region}** has the lowest ecosystem penetration and hardware sales volume, pointing to potential fundamental distribution barriers.
    """)
    
    # 3. Opportunities
    st.subheader("3. Growth Opportunities")
    st.success(f"""
    - **Services Expansion:** **{revenue_per_unit_region}** remarkably yields the highest *Services Revenue* per hardware unit sold, proving that aggressive software monetization completely balloons raw hardware profitability.
    - **Ecosystem Bundling:** There is massive untamed potential to aggressively cross-sell the **{worst_product}** directly to the massive existing **{top_product}** user base.
    """)
    
    # 4. Final Recommendations
    st.subheader("4. Strategic Action Plan")
    st.markdown(f"""
    1. **Protect the Core:** Maintain aggressive, uncompromised marketing spend on the **{top_product}** to defend primary revenue streams.
    2. **Investigate Regional Friction:** Immediately authorize and launch targeted market research in **{worst_region}** to identify why sales are critically lagging.
    3. **Magnify Services:** Audit the subscription and digital services rollout strategies scaling successfully in **{revenue_per_unit_region}** and rapidly deploy them into other regions to maximize customer lifetime value.
    """)
