"""
Model training module for Ontario house price prediction.

This module handles training of machine learning models (Linear Regression, Random Forest)
and saving trained models to disk for deployment.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

from src.data import (
    load_data, clean_data, engineer_features,
    prepare_features, split_data, save_encoders
)


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_regularization: bool = False,
    alpha: float = 1.0
) -> LinearRegression:
    """
    Train a Linear Regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    use_regularization : bool
        If True, use Ridge regression (L2 regularization)
    alpha : float
        Regularization strength (only used if use_regularization=True)
        
    Returns:
    --------
    LinearRegression or Ridge
        Trained model
    """
    if use_regularization:
        model = Ridge(alpha=alpha, random_state=42)
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Train a Random Forest Regressor model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_estimators : int
        Number of trees in the forest
    max_depth : int, optional
        Maximum depth of trees
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    RandomForestRegressor
        Trained model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> GradientBoostingRegressor:
    """
    Train a Gradient Boosting Regressor model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    n_estimators : int
        Number of boosting stages
    max_depth : int
        Maximum depth of individual trees
    learning_rate : float
        Learning rate (shrinkage)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    GradientBoostingRegressor
        Trained model
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Evaluate model performance on training and test sets.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    return metrics


def get_feature_importance(
    model,
    feature_names: list
) -> pd.DataFrame:
    """
    Get feature importance from tree-based models.
    
    Parameters:
    -----------
    model : sklearn model
        Trained tree-based model (RandomForest, GradientBoosting)
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with features and their importance scores, sorted by importance
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    return importance_df


def save_model(model, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    filepath : str
        Path to save the model
    """
    joblib.dump(model, filepath)


def load_model(filepath: str):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved model
        
    Returns:
    --------
    sklearn model
        Loaded model
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at {filepath}")
    return joblib.load(filepath)


def train_all_models(
    data_path: str = "properties.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    use_scaling: bool = False,
    lr_params: Optional[dict] = None,
    rf_params: Optional[dict] = None,
    save_models_flag: bool = True,
    models_dir: str = "models"
) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Complete training pipeline: load data, preprocess, train all models, evaluate, and save.
    
    Parameters:
    -----------
    data_path : str
        Path to the dataset CSV
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed for reproducibility
    use_scaling : bool
        Whether to apply feature scaling
    lr_params : dict, optional
        Parameters for Linear Regression
    rf_params : dict, optional
        Parameters for Random Forest
    save_models_flag : bool
        Whether to save models to disk
    models_dir : str
        Directory to save models
        
    Returns:
    --------
    Tuple containing:
        - models: dict of trained models
        - metrics: dict of evaluation metrics for each model
        - X_train, X_test, y_train, y_test
    """
    # Set default parameters
    if lr_params is None:
        lr_params = {'use_regularization': False}
    if rf_params is None:
        rf_params = {'n_estimators': 100, 'max_depth': 20}
    
    # Load and prepare data
    print("Loading data...")
    df = load_data(data_path)
    
    print("Cleaning data...")
    df = clean_data(df)
    
    print("Engineering features...")
    df = engineer_features(df)
    
    print("Preparing features...")
    X, encoders = prepare_features(
        df,
        fit_encoders=True,
        scaler=StandardScaler() if use_scaling else None
    )
    y = df['Price ($)']
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Train models
    models = {}
    metrics = {}
    
    print("\nTraining Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train, **lr_params)
    models['Linear Regression'] = lr_model
    metrics['Linear Regression'] = evaluate_model(lr_model, X_train, y_train, X_test, y_test)
    
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train, random_state=random_state, **rf_params)
    models['Random Forest'] = rf_model
    metrics['Random Forest'] = evaluate_model(rf_model, X_train, y_train, X_test, y_test)
    
    # Save models and encoders
    if save_models_flag:
        os.makedirs(models_dir, exist_ok=True)
        print(f"\nSaving models to {models_dir}/...")
        save_model(lr_model, f"{models_dir}/model_lr.joblib")
        save_model(rf_model, f"{models_dir}/model_rf.joblib")
        save_encoders(encoders, models_dir)
        print("Models and encoders saved successfully!")
    
    return models, metrics, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    """
    Train models with default settings when run as a script.
    """
    from sklearn.preprocessing import StandardScaler
    
    print("=" * 60)
    print("Ontario House Price Prediction - Model Training")
    print("=" * 60)
    
    models, metrics, X_train, X_test, y_train, y_test = train_all_models(
        data_path="properties.csv",
        use_scaling=True,
        rf_params={'n_estimators': 100, 'max_depth': 20}
    )
    
    print("\n" + "=" * 60)
    print("Model Performance Summary")
    print("=" * 60)
    
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        print(f"  Training RMSE: ${model_metrics['train_rmse']:,.2f}")
        print(f"  Test RMSE:     ${model_metrics['test_rmse']:,.2f}")
        print(f"  Training MAE:  ${model_metrics['train_mae']:,.2f}")
        print(f"  Test MAE:      ${model_metrics['test_mae']:,.2f}")
        print(f"  Training R²:   {model_metrics['train_r2']:.4f}")
        print(f"  Test R²:       {model_metrics['test_r2']:.4f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
