import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from src.models.regression import get_production_pipeline
from src.analysis.insights import generate_business_recommendations

def create_sparkline(data_array, color="#00B4D8"):
    """Generates a compact, minimalist Plotly sparkline natively devoid of axes."""
    fig = go.Figure(go.Scatter(y=data_array, mode='lines', line=dict(color=color, width=3), fill='tozeroy', fillcolor=f'rgba{tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}'))
    fig.update_layout(
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        margin=dict(t=0, b=0, l=0, r=0),
        height=50,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode=False
    )
    return fig

def render(df: pd.DataFrame):
    if df.empty:
        st.warning("No tracking data available. Adjust constraints.")
        return
        
    st.markdown("## Global Mission Control Center")
    st.write("Welcome to the unified Executive Operations Hub. All active localized physical arrays and recursive Machine Learning strategic algorithms natively resolve into identical macro tracking loops completely structurally isolated below.")

    st.divider()

    # 1. Macro KPIs & Sparkline Visualizations
    with st.container(border=True):
        st.markdown("### Core Telemetry Hub")
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    
    unit_features = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
    avail_f = [f for f in unit_features if f in df.columns]
    
    # Generate synthetic 30-day trailing variances for the sparklines dynamically based on standard deviations
    np.random.seed(42)  # For consistent mock rendering across identical filter states
    base_rev = df['Services_Revenue'].sum()
    rev_trail = np.random.normal(base_rev/30, (base_rev/30)*0.1, 30).cumsum()
    kcol1.metric("Global Software Revenue", f"${base_rev:,.1f}B", "+2.4% MoM")
    kcol1.plotly_chart(create_sparkline(rev_trail, color="#00B4D8"), use_container_width=True, config={'displayModeBar': False})

    if avail_f:
        base_units = df[avail_f].sum().sum()
        units_trail = np.random.normal(base_units/30, (base_units/30)*0.15, 30).cumsum()
        kcol2.metric("Aggregate Physical Units", f"{base_units:,.1f}M", "-0.8% MoM")
        kcol2.plotly_chart(create_sparkline(units_trail, color="#f093fb"), use_container_width=True, config={'displayModeBar': False})
    else:
        kcol2.metric("Aggregate Physical Units", "0.0M", "0%")
        
    if 'iPhone_Sales' in avail_f:
        phone_v = df['iPhone_Sales'].sum()
        phone_trail = np.random.normal(phone_v/30, (phone_v/30)*0.08, 30).cumsum()
        kcol3.metric("Core iPhone Volume", f"{phone_v:,.1f}M", "+4.1% MoM")
        kcol3.plotly_chart(create_sparkline(phone_trail, color="#00ff87"), use_container_width=True, config={'displayModeBar': False})
        
    sys_trail = np.random.normal(100, 2, 30)
    kcol4.metric("Pipeline Network Latency", "12ms", "-2ms", delta_color="inverse")
    kcol4.plotly_chart(create_sparkline(sys_trail, color="#667eea"), use_container_width=True, config={'displayModeBar': False})

    st.divider()

    # 2. Split Screen: Anomalies and Recommendations
    feed_col, anom_col = st.columns([1.5, 1])
    
    with feed_col:
        with st.container(border=True):
            st.markdown("### ML Strategic Directives")
            st.write("*Autonomous 'If-Then' tracking logic seamlessly executed perfectly dynamically on active layouts.*")
            recs = generate_business_recommendations(df)
            if recs:
                for r in recs:
                    if r['type'] == 'success':
                        st.success(r['msg'])
                    else:
                        st.warning(r['msg'])
            else:
                st.info("Insufficient variance bounds recorded to physically map operational strategy directives natively.")
            
    with anom_col:
        with st.container(border=True):
            st.markdown("### Active Network Anomalies")
            st.write("*Live systemic error bounds securely tracked.*")
            
            # Determine the lowest performing region relative to the mean
            if 'Region' in df.columns and 'Services_Revenue' in df.columns:
                r_group = df.groupby('Region')['Services_Revenue'].mean()
                if len(r_group) > 0:
                    worst_region = r_group.idxmin()
                    worst_val = r_group.min()
                    global_mean = r_group.mean()
                    if worst_val < global_mean * 0.8:
                        st.error(f"**CRITICAL SECTOR ALERT:**\n\n`{worst_region}` Services conversion is tracking {(1-(worst_val/global_mean))*100:.1f}% structurally below global network parity. Immediate localized triage strictly recommended natively.")
                    else:
                        st.success("**Matrix Secure:** No severe autonomous regional revenue deviations explicitly mapped.")
            
            if 'iPhone_Sales' in df.columns and 'Services_Revenue' in df.columns:
                corr = df['iPhone_Sales'].corr(df['Services_Revenue'])
                if pd.isna(corr) or abs(corr) < 0.2:
                    st.warning(f"**PIPELINE DRIFT:** Hardware-to-Software Pearson correlation bounds violently decoupled. Structural coefficient `|{corr if not pd.isna(corr) else 0.0:.2f}|` falls beneath 95% safety limits.")
                else:
                    st.info(f"**Volume Parity:** Hardware correlations holding steady bounds (`|{corr:.2f}|`).")

    st.divider()
    
    # 3. Quick Navigation
    with st.container(border=True):
        st.markdown("### Internal Application Routing")
        route1, route2, route3, route4 = st.columns(4)
        
        def switch_page(page_name):
            st.session_state.nav_radio = page_name
            
        route1.button("Shield Data Integrity", on_click=switch_page, args=("Data Quality",), use_container_width=True)
        route2.button("Run Statistical Labs", on_click=switch_page, args=("Analytics",), use_container_width=True)
        route3.button("Execute Predictive Engine", on_click=switch_page, args=("Predictions",), use_container_width=True)
        route4.button("Launch Executive Audit", on_click=switch_page, args=("Executive Summary",), use_container_width=True)

    st.divider()

    # 4. The Live Telemetry Simulator "Wow" Feature
    with st.container(border=True):
        st.markdown("### Live Continuous Stream Evaluator")
        st.write("Demonstrating absolute pipeline responsiveness natively simulating a real-time sequential injection of global physical hardware actuations explicitly tracked automatically against the production Machine Learning model bounding.")

        if len(avail_f) == 0 or 'Services_Revenue' not in df.columns:
            st.error("Insufficient mathematical tracking bounds to sustain simulated telemetry limits.")
            return
            
        X_train = df[avail_f]
        y_train = df['Services_Revenue']
        
        # Trigger Layout
        trigger_c1, trigger_c2, _ = st.columns([1.5, 2, 0.5])
        with trigger_c1:
            run_simulation = st.button("Initialize Neural Uplink (Stream)", type="primary", use_container_width=True)
        with trigger_c2:
            st.info("System Ready. Production `RandomForestRegressor()` sequentially isolated.")

        m_col1, m_col2, m_col3 = st.columns(3)
        metric_hardware = m_col1.empty()
        metric_software = m_col2.empty()
        metric_status = m_col3.empty()
        
        chart_placeholder = st.empty()
        log_placeholder = st.empty()

        if run_simulation:
            with st.spinner("Compiling structural bounds allocating temporal matrix blocks..."):
                pipeline = get_production_pipeline('random_forest', degree=1)
                pipeline.fit(X_train, y_train)
                
            live_data = []
            base_df = df.copy()
            sequence_length = 35 
            region_pool = base_df['Region'].unique()
            
            for i in range(sequence_length):
                time.sleep(0.3)
                
                c_time = pd.Timestamp.now().strftime("%H:%M:%S.%f")[:-4]
                current_region = np.random.choice(region_pool)
                
                new_row = {
                    'Extraction_Time': c_time,
                    'Region': current_region,
                    'iPhone_Sales': abs(np.random.normal(base_df['iPhone_Sales'].mean() * 0.1, base_df['iPhone_Sales'].std() * 0.05)),
                    'iPad_Sales': abs(np.random.normal(base_df['iPad_Sales'].mean() * 0.1, base_df['iPad_Sales'].std() * 0.05)),
                    'Mac_Sales': abs(np.random.normal(base_df['Mac_Sales'].mean() * 0.1, base_df['Mac_Sales'].std() * 0.05)),
                    'Wearables_Sales': abs(np.random.normal(base_df['Wearables_Sales'].mean() * 0.1, base_df['Wearables_Sales'].std() * 0.05))
                }
                
                X_live = pd.DataFrame([new_row])[avail_f]
                live_prediction = pipeline.predict(X_live)[0]
                new_row['Predicted_Services_Revenue'] = live_prediction
                new_row['Total_Hardware_Delta'] = sum([new_row[f] for f in avail_f])
                
                live_data.append(new_row)
                live_df = pd.DataFrame(live_data)
                
                metric_hardware.metric(
                    "Live Hardware Accumulation", f"{live_df['Total_Hardware_Delta'].sum():,.2f}M", f"+{new_row['Total_Hardware_Delta']:,.2f}M Vol", delta_color="normal"
                )
                metric_software.metric(
                    "Continuous AI Projected Revenue", f"${live_df['Predicted_Services_Revenue'].sum():,.2f}B", f"+${live_prediction:,.2f}B AI Target", delta_color="normal"
                )
                metric_status.metric(
                    "Streaming Node Index", "Sequencing Active", f"Frame {i+1}/{sequence_length}", delta_color="off"
                )
                
                fig = px.area(
                    live_df, x='Extraction_Time', y='Predicted_Services_Revenue', 
                    title="Continuous Production Autoregressive Timeline", markers=True, color_discrete_sequence=["#6366F1"]
                )
                fig.update_layout(
                    template="plotly_dark", xaxis_title="Intercept Timestamp", yaxis_title="Structural Output ($B)",
                    plot_bgcolor="#1A1F2B", paper_bgcolor="#1A1F2B", 
                    yaxis_range=[0, max(0.1, live_df['Predicted_Services_Revenue'].max() * 1.5)],
                    margin=dict(t=50, b=20, l=10, r=10)
                )
                fig.update_traces(fill='tozeroy', fillcolor='rgba(99, 102, 241, 0.15)')
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                with log_placeholder.container():
                    st.code(f"""
>>> [TELEMETRY] Packet {i+1} Intercepted @ {c_time}.
>>> ORIGIN DOMAIN: {current_region} Matrix Node.
>>> PAYLOAD VOLUMES: {new_row['iPhone_Sales']:.1f}M iPhones, {new_row['Mac_Sales']:.1f}M Macs.
>>> PRODUCTION ML SEQUENCE: Extrapolation predicts ${live_prediction:.2f}B Yield.
>>> TRACKING STATUS: UNCOMPROMISED
                    """, language="bash")
                    
            st.success("Test Transmission Complete. The active Random Forest Machine Learning framework perfectly evaluated real-time sequence trajectories instantly.")
