import streamlit as st

def initialize_demo():
    """Binds fundamental tracking parameters securely into the session state matrix."""
    if 'demo_active' not in st.session_state:
        st.session_state.demo_active = False
        st.session_state.demo_step = 0

def start_demo():
    """Callback explicitly triggering the sequential override array natively."""
    st.session_state.demo_active = True
    st.session_state.demo_step = 0
    st.session_state.nav_radio = "Overview"

def next_demo_step():
    """Callback physically advancing the state sequence."""
    steps = ["Overview", "Analytics", "Predictions", "Live Mission Control"]
    st.session_state.demo_step += 1
    if st.session_state.demo_step < len(steps):
        st.session_state.nav_radio = steps[st.session_state.demo_step]
    else:
        st.session_state.demo_active = False
        st.session_state.nav_radio = "Live Mission Control"

def exit_demo():
    """Callback securely abandoning the override sequence."""
    st.session_state.demo_active = False
    st.session_state.nav_radio = "Live Mission Control"

def render_demo_overlay():
    """Renders the physical presentation logic securely tracking above standard module rendering."""
    if st.session_state.get('demo_active', False):
        steps = ["Overview", "Analytics", "Predictions", "Live Mission Control"]
        
        # Failsafe limit
        if st.session_state.demo_step >= len(steps):
            st.session_state.demo_active = False
            return
            
        current_step_name = steps[st.session_state.demo_step]
        
        # Enforce strict identical application routing to match the demo step precisely
        if st.session_state.nav_radio != current_step_name:
            st.session_state.nav_radio = current_step_name
            st.rerun()
            
        st.markdown(f"### **Guided Pitch Mode ({st.session_state.demo_step + 1}/{len(steps)}): {current_step_name}**")
        
        if current_step_name == "Overview":
            st.info("*Start here to view the macro landscape. Notice exactly how raw dataset nodes are natively parsed into top-level KPIs effortlessly while the C-Suite AI engine autonomously prints localized Strategic Directives based entirely on structural state distributions.*")
        elif current_step_name == "Analytics":
            st.info("*Next, rigorous statistical evaluations are independently applied. Observe the internal `ANOVA` and `Causal Inference` blocks mathematically generating literal $USD action constraints dynamically bypassing pure correlation tracking securely.*")
        elif current_step_name == "Predictions":
            st.info("*Now, algorithms predict absolute physical future topologies. Scroll down to the `Optimization Engine` to literally watch the `Scipy SLSQP` combinations min-max exact regional revenue bounds perfectly securing maximum mathematical yields natively.*")
        elif current_step_name == "Live Mission Control":
            st.info("*Finally, structural validation globally concludes back at Mission Control. Click **Initialize Neural Uplink (Stream)** below to watch the underlying `RandomForestRegressor` evaluate fully raw synthetic live streaming matrices absolutely instantaneously!*")
            
        col1, col2, _ = st.columns([1, 1, 4])
        if st.session_state.demo_step < len(steps) - 1:
            col1.button("Next Presentation Layout", on_click=next_demo_step, type="primary")
        else:
            col1.button("Finish Demo Run", on_click=exit_demo, type="primary")
        
        col2.button("Exit Demo Mode", on_click=exit_demo)
        st.divider()
