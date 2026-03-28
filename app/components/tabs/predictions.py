import streamlit as st
import pandas as pd
from src.models.regression import simple_linear_regression, multiple_linear_regression, polynomial_regression, get_production_pipeline
from src.visualization.charts import plot_actual_vs_predicted, plot_feature_importance, plot_residuals
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def render(df: pd.DataFrame):
    st.markdown("### Automated Feature Selection")
    st.write("Evaluating linear correlation thresholds and non-linear importance hierarchies to dynamically prune useless mathematical signals and multi-collinearity prior to active model training.")
    
    y = df['Services_Revenue']
    base_features = ['iPhone_Sales', 'iPad_Sales', 'Mac_Sales', 'Wearables_Sales']
    
    # 1. Linear Correlation Filter
    correlations = df[base_features].corrwith(y).abs()
    
    # 2. Non-Linear Signal Filter (Random Forest baseline)
    rf_eval = RandomForestRegressor(random_state=42, n_estimators=50)
    rf_eval.fit(df[base_features], y)
    importances = pd.Series(rf_eval.feature_importances_, index=base_features)
    
    # Structural Matrix Filtering Logic
    features = []
    dropped = []
    
    for f in base_features:
        # Retention heuristics: Significant correlation, OR significant structural splitting importance
        if correlations[f] > 0.05 or importances[f] > 0.10:
            features.append(f)
        else:
            dropped.append(f)
            
    # Rescue logic: If completely zero variance, isolate the most statistically powerful node
    if not features:
        highest_signal = importances.idxmax()
        features = [highest_signal]
        if highest_signal in dropped:
            dropped.remove(highest_signal)
            
    st.success(f"**Retained Strong Features**: `{', '.join(features)}`")
    if dropped:
        st.warning(f"**Pruned Weak Features** (Mathematically Insignificant): `{', '.join(dropped)}`")
        
    st.divider()

    model_type = st.selectbox("Select Predictive Model Type", [
        "Linear Regression", 
        "Multiple Linear Regression", 
        "Polynomial Features",
        "Random Forest (Non-Linear)",
        "Gradient Boosting (Non-Linear)"
    ])
    
    st.divider()
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if model_type == "Linear Regression":
            single_feature = features[0] # Single feature linear locks to the single most powerful surviving node
            model, y_pred, r2, mse, rmse = simple_linear_regression(df, single_feature, 'Services_Revenue')
            
        elif model_type == "Multiple Linear Regression":
            model, y_pred, r2, mse, rmse = multiple_linear_regression(df, features, 'Services_Revenue')
            
        elif model_type == "Polynomial Features":
            poly_degree = st.slider("Polynomial Degree", 1, 5, 2)
            model, y_pred, r2, mse, rmse = polynomial_regression(df, features, 'Services_Revenue', poly_degree)
            
        elif "Forest" in model_type or "Boosting" in model_type:
            X = df[features]
            m_type = 'random_forest' if 'Forest' in model_type else 'gradient_boosting'
            pipeline = get_production_pipeline(m_type, degree=1)
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)
            r2 = r2_score(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            model = pipeline
            
        st.markdown("### Model Visualizations")
        fig_actual = plot_actual_vs_predicted(y, y_pred)
        st.plotly_chart(fig_actual, use_container_width=True)
        
        fig_res = plot_residuals(y_pred, y - y_pred)
        st.plotly_chart(fig_res, use_container_width=True)
        
        if model_type in ["Multiple Linear Regression", "Random Forest (Non-Linear)", "Gradient Boosting (Non-Linear)"]:
            coefs = model.coef_ if model_type == "Multiple Linear Regression" else model.named_steps['model'].feature_importances_
            fig_imp = plot_feature_importance(features, coefs)
            st.plotly_chart(fig_imp, use_container_width=True)
            
    with col2:
        mae = mean_absolute_error(y, y_pred)
        
        st.markdown("### Model Performance")
        st.metric(label="R² Score", value=f"{r2:.3f}")
        st.metric(label="RMSE", value=f"{rmse:.2f}")
        st.metric(label="MAE", value=f"{mae:.2f}")

        st.divider()
        st.markdown("### Sales Simulator")
        st.write("Adjust retained product lines (M units) to simulate the predicted **Services Revenue**.")

        if model_type == "Linear Regression":
            sim_input = st.slider(f"{single_feature} (M)", min_value=0.0, max_value=200.0, value=25.0, step=1.0)
            input_df = pd.DataFrame([[sim_input]], columns=[single_feature])
            pred = model.predict(input_df)[0]
            st.metric("Predicted Services Revenue", f"${pred:.2f}B")
            
        elif model_type in ["Multiple Linear Regression", "Random Forest (Non-Linear)", "Gradient Boosting (Non-Linear)", "Polynomial Features"]:
            input_dict = {}
            for f in features:
                if 'iPhone' in f: input_dict[f] = st.slider("iPhone Sales (M)", 0.0, 200.0, 25.0, step=1.0)
                elif 'iPad' in f: input_dict[f] = st.slider("iPad Sales (M)", 0.0, 100.0, 10.0, step=1.0)
                elif 'Mac' in f: input_dict[f] = st.slider("Mac Sales (M)", 0.0, 100.0, 8.0, step=1.0)
                elif 'Wearables' in f: input_dict[f] = st.slider("Wearables Sales", 0.0, 100.0, 12.0, step=1.0)
            
            input_df = pd.DataFrame([input_dict])
            pred = model.predict(input_df)[0]
            st.metric("Predicted Services Revenue", f"${pred:.2f}B")
