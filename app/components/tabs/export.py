import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import datetime
import io
from src.models.regression import get_production_pipeline

def render(df: pd.DataFrame):
    if df.empty:
        st.warning("No tracking data available. Adjust constraints.")
        return
        
    st.markdown("## Persistence & Reproducibility Hub")
    st.write("Securely deploy and export localized autonomous machine learning bounds. Download raw mathematical inference matrices internally mapped identically to production environments, strictly adhering to pipeline reproducibility protocols.")
    
    st.divider()

    features = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
    avail_f = [f for f in features if f in df.columns]
    y_col = 'Services_Revenue'
    
    if len(avail_f) == 0 or y_col not in df.columns:
        st.error("Insufficient topological features required to execute persistent export bounds natively.")
        return

    X = df[avail_f]
    y = df[y_col]
    
    with st.spinner("Compiling production objects into isolated sequential memory states..."):
        # 1. Train Production Model securely
        model = get_production_pipeline('random_forest', degree=1)
        model.fit(X, y)
        
        # Determine internal limits mathematically caching objects
        os.makedirs("src/models/saved", exist_ok=True)
        model_path = "src/models/saved/production_rf_model_v1.joblib"
        joblib.dump(model, model_path)

        # 2. Build JSON Config
        hyper_params = model.named_steps['model'].get_params()
        config_payload = {
            "System_Architecture": "Apple Intelligence Pipeline v1.0",
            "Extraction_Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Target_Dependency": "Services_Revenue",
            "Validated_Features": avail_f,
            "Internal_Variance_Nodes": len(df),
            "Active_Algorithms": {
                "Scaling": "StandardScaler(with_mean=True, with_std=True)",
                "Estimator": "RandomForestRegressor()"
            },
            "Hyperparameters_Locked": hyper_params
        }
        json_output = json.dumps(config_payload, indent=4)
        
        # 3. Compile Forecasting Predictions CSV
        scenario_df = X.copy()
        # Add a "Baseline Validation" forecast natively mapping outputs explicitly
        scenario_df['Projected_Services_Revenue'] = model.predict(X)
        scenario_df['Residual_Error_Delta'] = scenario_df['Projected_Services_Revenue'] - y
        scenario_df['Absolute_Physical_Target'] = y
        
        csv_buffer = io.BytesIO()
        scenario_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
    # Layout the UI blocks perfectly
    ecol1, ecol2, ecol3 = st.columns(3)
    
    with ecol1:
        with st.container(border=True):
            st.markdown("### 1. Binary Model Object")
            st.write("Download the literal serialized Python native `StandardScaler` + `Random Forest` production pipeline directly. Executable globally inside any identical `scikit-learn` ecosystem natively.")
            st.caption("`.joblib` Object format.")
            
            with open(model_path, "rb") as f:
                st.download_button(
                    label="Download Production Model",
                    data=f,
                    file_name="apple_production_rf_v1.joblib",
                    mime="application/octet-stream",
                    type="primary",
                    use_container_width=True,
                    help="Exports the globally optimized Random Forest algorithmic limits completely intact natively preserving all StandardScaler gradients."
                )

    with ecol2:
        with st.container(border=True):
            st.markdown("### 2. Reproducibility Configuration")
            st.write("Export a static topological dictionary expressly dictating active hyper-parameters, pipeline boundaries, and temporal environmental context mapping completely autonomously.")
            st.caption("`.json` State format.")
            
            st.download_button(
                label="Download Pipeline State",
                data=json_output,
                file_name="pipeline_reproducibility_config.json",
                mime="application/json",
                use_container_width=True,
                help="Exports explicit JSON formatting exposing tree depths, standard estimators, and execution node arrays sequentially."
            )
            
    with ecol3:
        with st.container(border=True):
            st.markdown("### 3. Baseline Validation Forecasts")
            st.write("Extract the explicit algorithmic performance constraints mapping physical matrix truths natively to expected sequential projection limits.")
            st.caption("`.csv` Spreadsheet format.")
            
            st.download_button(
                label="Download Forecasting Report",
                data=csv_buffer.getvalue(),
                file_name="baseline_validation_forecast.csv",
                mime="text/csv",
                use_container_width=True,
                help="Generates an exact spreadsheet defining tracking coordinates natively bridging Projected vs Actual targets seamlessly."
            )

    st.divider()
    
    # Render Audit payload natively natively
    with st.container(border=True):
        st.markdown("### Internal State Audit")
        with st.expander("Expand to view locally compiled Reproducibility Dictionary bindings."):
            st.json(config_payload)
