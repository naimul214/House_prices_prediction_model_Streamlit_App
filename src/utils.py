"""
Utility functions for the Ontario house price prediction application.

This module contains helper functions for validation, formatting,
and common operations used across the application.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import re


def validate_csv_upload(df: pd.DataFrame) -> Tuple[bool, str, List[str]]:
    """
    Validate that an uploaded CSV has the required columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Uploaded dataframe
        
    Returns:
    --------
    Tuple[bool, str, List[str]]
        (is_valid, error_message, missing_columns)
    """
    required_columns = ['Address', 'AreaName', 'lat', 'lng']
    optional_columns = ['Price ($)']
    
    missing_required = [col for col in required_columns if col not in df.columns]
    
    if missing_required:
        return False, f"Missing required columns: {', '.join(missing_required)}", missing_required
    
    return True, "", []


def format_currency(value: float) -> str:
    """
    Format a numeric value as currency.
    
    Parameters:
    -----------
    value : float
        Numeric value to format
        
    Returns:
    --------
    str
        Formatted currency string
    """
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """
    Format a numeric value as percentage.
    
    Parameters:
    -----------
    value : float
        Numeric value to format (e.g., 0.85 for 85%)
        
    Returns:
    --------
    str
        Formatted percentage string
    """
    return f"{value * 100:.2f}%"


def calculate_price_range(price: float, margin: float = 0.10) -> Tuple[float, float]:
    """
    Calculate a price range around a predicted price.
    
    Parameters:
    -----------
    price : float
        Predicted price
    margin : float
        Margin percentage (default: 0.10 for ±10%)
        
    Returns:
    --------
    Tuple[float, float]
        (lower_bound, upper_bound)
    """
    margin_amount = price * margin
    lower_bound = price - margin_amount
    upper_bound = price + margin_amount
    return lower_bound, upper_bound


def extract_city_from_address(address: str) -> str:
    """
    Extract city name from a full address string.
    
    Parameters:
    -----------
    address : str
        Full address string (e.g., "123 Main St Toronto, ON")
        
    Returns:
    --------
    str
        Extracted city name
    """
    # Pattern to match city before ", ON"
    match = re.search(r'([A-Za-z\s]+),\s*ON', address)
    if match:
        return match.group(1).strip()
    return "Unknown"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Parameters:
    -----------
    filename : str
        Original filename
        
    Returns:
    --------
    str
        Sanitized filename
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    return sanitized


def get_area_statistics(df: pd.DataFrame, area_name: str) -> dict:
    """
    Calculate statistics for a specific area.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with property information
    area_name : str
        Name of the area
        
    Returns:
    --------
    dict
        Dictionary with area statistics
    """
    area_data = df[df['AreaName'] == area_name]
    
    if len(area_data) == 0:
        return {
            'count': 0,
            'avg_price': None,
            'median_price': None,
            'min_price': None,
            'max_price': None
        }
    
    stats = {
        'count': len(area_data),
        'avg_price': area_data['Price ($)'].mean(),
        'median_price': area_data['Price ($)'].median(),
        'min_price': area_data['Price ($)'].min(),
        'max_price': area_data['Price ($)'].max()
    }
    
    return stats


def get_top_areas(df: pd.DataFrame, n: int = 10, by: str = 'avg_price') -> pd.DataFrame:
    """
    Get top N areas by a specific metric.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with property information
    n : int
        Number of top areas to return
    by : str
        Metric to sort by ('avg_price', 'median_price', 'count')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with top areas and their statistics
    """
    area_stats = df.groupby('AreaName')['Price ($)'].agg([
        ('avg_price', 'mean'),
        ('median_price', 'median'),
        ('count', 'size')
    ]).reset_index()
    
    if by not in area_stats.columns:
        by = 'avg_price'
    
    top_areas = area_stats.nlargest(n, by)
    return top_areas


def detect_outliers(df: pd.DataFrame, column: str = 'Price ($)', method: str = 'iqr') -> pd.Series:
    """
    Detect outliers in a numeric column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    column : str
        Column to check for outliers
    method : str
        Method to use ('iqr' or 'zscore')
        
    Returns:
    --------
    pd.Series
        Boolean series indicating outliers (True = outlier)
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > 3
    
    return pd.Series([False] * len(df))


def create_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Create comprehensive summary statistics for the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
        
    Returns:
    --------
    dict
        Dictionary with summary statistics
    """
    summary = {
        'total_properties': len(df),
        'unique_areas': df['AreaName'].nunique(),
        'price_statistics': {
            'mean': df['Price ($)'].mean(),
            'median': df['Price ($)'].median(),
            'std': df['Price ($)'].std(),
            'min': df['Price ($)'].min(),
            'max': df['Price ($)'].max()
        }
    }
    
    if 'City' in df.columns:
        summary['unique_cities'] = df['City'].nunique()
    
    return summary


def generate_sample_data(df: pd.DataFrame, n: int = 10, seed: int = 42) -> pd.DataFrame:
    """
    Generate a sample dataset for demonstration.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    n : int
        Number of samples
    seed : int
        Random seed
        
    Returns:
    --------
    pd.DataFrame
        Sample dataset
    """
    return df.sample(n=min(n, len(df)), random_state=seed)


def validate_price(price: float) -> Tuple[bool, str]:
    """
    Validate that a price value is reasonable.
    
    Parameters:
    -----------
    price : float
        Price value to validate
        
    Returns:
    --------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    if price <= 0:
        return False, "Price must be positive"
    
    if price < 10000:
        return False, "Price seems too low (minimum $10,000)"
    
    if price > 50000000:
        return False, "Price seems too high (maximum $50,000,000)"
    
    return True, ""


def calculate_metrics_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate a comprehensive set of regression metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary with various metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    residuals = y_true - y_pred
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs(residuals / y_true)) * 100,  # Mean Absolute Percentage Error
        'median_error': np.median(np.abs(residuals))
    }
    
    return metrics


def create_results_dataframe(
    addresses: List[str],
    actual_prices: Optional[List[float]],
    predicted_prices: List[float],
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Create a formatted results dataframe for display.
    
    Parameters:
    -----------
    addresses : List[str]
        Property addresses
    actual_prices : List[float], optional
        Actual prices (if available)
    predicted_prices : List[float]
        Predicted prices
    confidence_intervals : List[Tuple[float, float]], optional
        Confidence intervals for predictions
        
    Returns:
    --------
    pd.DataFrame
        Formatted results dataframe
    """
    results = pd.DataFrame({
        'Address': addresses,
        'Predicted Price': [format_currency(p) for p in predicted_prices]
    })
    
    if actual_prices is not None:
        results['Actual Price'] = [format_currency(p) for p in actual_prices]
        errors = [pred - actual for pred, actual in zip(predicted_prices, actual_prices)]
        results['Difference'] = [format_currency(e) for e in errors]
        results['Error %'] = [f"{(e/actual)*100:.2f}%" for e, actual in zip(errors, actual_prices)]
    
    if confidence_intervals is not None:
        results['Confidence Interval'] = [
            f"{format_currency(ci[0])} - {format_currency(ci[1])}" 
            for ci in confidence_intervals
        ]
    
    return results


def get_color_for_error(error_percentage: float) -> str:
    """
    Get a color code based on prediction error percentage.
    
    Parameters:
    -----------
    error_percentage : float
        Absolute error percentage
        
    Returns:
    --------
    str
        Color code (green, yellow, red)
    """
    abs_error = abs(error_percentage)
    
    if abs_error < 5:
        return "green"
    elif abs_error < 10:
        return "yellow"
    else:
        return "red"
