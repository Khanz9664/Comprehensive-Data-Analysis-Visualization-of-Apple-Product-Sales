import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, GridSearchCV

def simple_linear_regression(df: pd.DataFrame, feature_col: str, target_col: str):
    """Trains a Simple Linear Regression model."""
    X = df[[feature_col]]
    y = df[target_col]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    return model, y_pred, r2, mse, rmse

def multiple_linear_regression(df: pd.DataFrame, feature_cols: list, target_col: str):
    """Trains a Multiple Linear Regression model."""
    X = df[feature_cols]
    y = df[target_col]
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    return model, y_pred, r2, mse, rmse

def polynomial_regression(df: pd.DataFrame, feature_cols: list, target_col: str, degree: int = 2):
    """Trains a Polynomial Regression model."""
    if isinstance(feature_cols, str):
        feature_cols = [feature_cols]
    if isinstance(feature_cols, list) and len(feature_cols) == 1 and isinstance(feature_cols[0], tuple):
        feature_cols = list(feature_cols[0])
    
    X = df[list(feature_cols)]
    y = df[target_col]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('linear', LinearRegression())
    ])
    
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    return pipeline, y_pred, r2, mse, rmse

def get_production_pipeline(model_type='linear', degree=2):
    """Returns a production-grade regression pipeline with scaling."""
    steps = [
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False))
    ]
    
    if model_type == 'random_forest':
        steps.append(('model', RandomForestRegressor(random_state=42)))
    elif model_type == 'gradient_boosting':
        steps.append(('model', GradientBoostingRegressor(random_state=42)))
    else:
        steps.append(('model', LinearRegression()))
        
    return Pipeline(steps)

def compare_models_cv(df: pd.DataFrame, feature_cols: list, target_col: str, cv_splits: int = 5):
    """Compares multiple regression models using KFold Cross-Validation and GridSearchCV."""
    X = df[feature_cols]
    y = df[target_col]
    
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    # Define models and their hyperparameter grids to search over
    models_to_test = {
        'Linear Regression': {
            'pipeline': get_production_pipeline('linear'),
            'params': {'poly__degree': [1, 2, 3]}
        },
        'Random Forest': {
            'pipeline': get_production_pipeline('random_forest', degree=1),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 5, 10, 20]
            }
        },
        'Gradient Boosting': {
            'pipeline': get_production_pipeline('gradient_boosting', degree=1),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2]
            }
        }
    }
    
    results = {}
    
    for name, config in models_to_test.items():
        grid_search = GridSearchCV(
            config['pipeline'], 
            config['params'], 
            cv=kf, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        
        # Calculate RMSE metric from negative MSE
        best_rmse = np.sqrt(-grid_search.best_score_)
        
        results[name] = {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'cv_rmse': best_rmse
        }
        
    return results
