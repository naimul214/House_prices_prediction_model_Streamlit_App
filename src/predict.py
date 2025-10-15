"""
Prediction and inference module for Ontario house price prediction.

This module handles making predictions with trained models and
preparing input data for inference.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import joblib

from src.data import engineer_features, prepare_features


def load_model_and_encoders(
    model_path: str,
    encoders_dir: str = "models"
) -> Tuple[object, dict]:
    """
    Load a trained model and its associated encoders.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    encoders_dir : str
        Directory containing encoder files
        
    Returns:
    --------
    Tuple[object, dict]
        Loaded model and dictionary of encoders
    """
    model = joblib.load(model_path)
    
    # Load encoders
    encoders = {}
    encoder_names = ['area_encoder', 'city_encoder', 'scaler']
    
    for name in encoder_names:
        filepath = f"{encoders_dir}/{name}.joblib"
        try:
            encoders[name] = joblib.load(filepath)
        except FileNotFoundError:
            if name == 'scaler':
                encoders[name] = None  # Scaler is optional
            else:
                raise FileNotFoundError(f"Encoder {name} not found at {filepath}")
    
    return model, encoders


def prepare_single_input(
    address: str,
    area_name: str,
    lat: float,
    lng: float,
    encoders: dict,
    reference_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Prepare a single property input for prediction.
    
    Parameters:
    -----------
    address : str
        Property address
    area_name : str
        Area/neighborhood name
    lat : float
        Latitude
    lng : float
        Longitude
    encoders : dict
        Dictionary of fitted encoders
    reference_df : pd.DataFrame, optional
        Reference dataframe for calculating area statistics
        
    Returns:
    --------
    pd.DataFrame
        Prepared features ready for prediction
    """
    # Create input dataframe
    input_data = pd.DataFrame({
        'Address': [address],
        'AreaName': [area_name],
        'lat': [lat],
        'lng': [lng],
        'Price ($)': [0]  # Dummy value, not used for prediction
    })
    
    # Engineer features
    input_data = engineer_features(input_data)
    
    # If reference data provided, use it for area statistics
    if reference_df is not None:
        # Calculate area average price from reference data
        area_avg = reference_df[reference_df['AreaName'] == area_name]['Price ($)'].mean()
        if pd.notna(area_avg):
            input_data['Area_Avg_Price'] = area_avg
        
        # Calculate area property count
        area_count = len(reference_df[reference_df['AreaName'] == area_name])
        if area_count > 0:
            input_data['Area_Property_Count'] = area_count
    
    # Prepare features with encoders
    X, _ = prepare_features(
        input_data,
        fit_encoders=False,
        label_encoder_area=encoders['area_encoder'],
        label_encoder_city=encoders['city_encoder'],
        scaler=encoders.get('scaler')
    )
    
    return X


def predict_price(
    model,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Make price predictions using a trained model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X : pd.DataFrame
        Prepared features
        
    Returns:
    --------
    np.ndarray
        Predicted prices
    """
    predictions = model.predict(X)
    return predictions


def predict_with_confidence(
    model,
    X: pd.DataFrame,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with confidence intervals (for Random Forest models).
    
    Parameters:
    -----------
    model : sklearn model
        Trained model (preferably Random Forest)
    X : pd.DataFrame
        Prepared features
    confidence_level : float
        Confidence level for intervals (default: 0.95)
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        predictions, lower_bounds, upper_bounds
    """
    predictions = model.predict(X)
    
    # For Random Forest, use tree predictions for confidence intervals
    if hasattr(model, 'estimators_'):
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
        
        # Calculate percentiles for confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(tree_predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(tree_predictions, upper_percentile, axis=0)
    else:
        # For models without tree ensembles, use a simple heuristic
        # (e.g., ±10% of prediction)
        margin = predictions * 0.10
        lower_bounds = predictions - margin
        upper_bounds = predictions + margin
    
    return predictions, lower_bounds, upper_bounds


def batch_predict(
    model,
    encoders: dict,
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Make predictions for a batch of properties.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    encoders : dict
        Dictionary of fitted encoders
    df : pd.DataFrame
        Dataframe with properties to predict
    reference_df : pd.DataFrame, optional
        Reference dataframe for area statistics
        
    Returns:
    --------
    pd.DataFrame
        Input dataframe with added prediction columns
    """
    df_pred = df.copy()
    
    # Engineer features
    df_pred = engineer_features(df_pred)
    
    # Update area statistics if reference provided
    if reference_df is not None:
        ref_eng = engineer_features(reference_df)
        for area in df_pred['AreaName'].unique():
            area_mask = df_pred['AreaName'] == area
            ref_mask = ref_eng['AreaName'] == area
            
            if ref_mask.any():
                df_pred.loc[area_mask, 'Area_Avg_Price'] = ref_eng.loc[ref_mask, 'Area_Avg_Price'].iloc[0]
                df_pred.loc[area_mask, 'Area_Property_Count'] = ref_eng.loc[ref_mask, 'Area_Property_Count'].iloc[0]
    
    # Prepare features
    X, _ = prepare_features(
        df_pred,
        fit_encoders=False,
        label_encoder_area=encoders['area_encoder'],
        label_encoder_city=encoders['city_encoder'],
        scaler=encoders.get('scaler')
    )
    
    # Make predictions
    predictions = predict_price(model, X)
    df_pred['Predicted_Price'] = predictions
    
    # Calculate difference if actual price exists
    if 'Price ($)' in df_pred.columns:
        df_pred['Prediction_Error'] = df_pred['Predicted_Price'] - df_pred['Price ($)']
        df_pred['Absolute_Error'] = df_pred['Prediction_Error'].abs()
        df_pred['Percentage_Error'] = (df_pred['Prediction_Error'] / df_pred['Price ($)']) * 100
    
    return df_pred


def validate_input(
    address: str,
    area_name: str,
    lat: float,
    lng: float
) -> Tuple[bool, str]:
    """
    Validate user input for predictions.
    
    Parameters:
    -----------
    address : str
        Property address
    area_name : str
        Area/neighborhood name
    lat : float
        Latitude
    lng : float
        Longitude
        
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Check address
    if not address or len(address.strip()) == 0:
        return False, "Address cannot be empty"
    
    # Check area name
    if not area_name or len(area_name.strip()) == 0:
        return False, "Area name cannot be empty"
    
    # Validate latitude (Ontario range: approximately 41-57°N)
    if not (41.0 <= lat <= 57.0):
        return False, f"Latitude {lat} is outside Ontario bounds (41-57°N)"
    
    # Validate longitude (Ontario range: approximately -95 to -74°W)
    if not (-95.0 <= lng <= -74.0):
        return False, f"Longitude {lng} is outside Ontario bounds (-95 to -74°W)"
    
    return True, ""


def get_feature_names() -> List[str]:
    """
    Get the list of feature names used by the models.
    
    Returns:
    --------
    List[str]
        List of feature names
    """
    return [
        'lat', 'lng', 'Distance_From_Downtown', 'Area_Avg_Price',
        'Area_Property_Count', 'AreaName_Encoded', 'City_Encoded'
    ]


def format_prediction_output(
    predictions: np.ndarray,
    lower_bounds: Optional[np.ndarray] = None,
    upper_bounds: Optional[np.ndarray] = None
) -> List[Dict[str, Union[float, str]]]:
    """
    Format predictions for display in the UI.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Predicted prices
    lower_bounds : np.ndarray, optional
        Lower confidence bounds
    upper_bounds : np.ndarray, optional
        Upper confidence bounds
        
    Returns:
    --------
    List[Dict[str, Union[float, str]]]
        Formatted predictions with confidence intervals
    """
    results = []
    
    for i, pred in enumerate(predictions):
        result = {
            'predicted_price': float(pred),
            'formatted_price': f"${pred:,.2f}"
        }
        
        if lower_bounds is not None and upper_bounds is not None:
            result['confidence_interval'] = f"${lower_bounds[i]:,.2f} - ${upper_bounds[i]:,.2f}"
            result['lower_bound'] = float(lower_bounds[i])
            result['upper_bound'] = float(upper_bounds[i])
        
        results.append(result)
    
    return results
