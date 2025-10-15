"""
Data loading and preprocessing module for Ontario house price prediction.

This module handles dataset loading, feature engineering, preprocessing,
and train-test splitting for the house price prediction models.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def load_data(filepath: str = "properties.csv") -> pd.DataFrame:
    """
    Load the Ontario housing dataset from CSV.
    
    Parameters:
    -----------
    filepath : str
        Path to the properties CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset with all columns
    """
    try:
        df = pd.read_csv(filepath)
        
        # Drop the unnamed index column if it exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing invalid entries and handling missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset
    """
    df_clean = df.copy()
    
    # Remove entries with invalid prices (< $1000 or unrealistically high)
    # Some entries have prices like 25, 49, 97, 113, 145 which seem like errors
    df_clean = df_clean[df_clean['Price ($)'] >= 1000]
    df_clean = df_clean[df_clean['Price ($)'] <= 10000000]  # Max 10M
    
    # Remove rows with missing critical values
    df_clean = df_clean.dropna(subset=['Address', 'AreaName', 'Price ($)'])
    
    # Fill missing lat/lng with area averages
    for area in df_clean['AreaName'].unique():
        area_mask = df_clean['AreaName'] == area
        df_clean.loc[area_mask, 'lat'] = df_clean.loc[area_mask, 'lat'].fillna(
            df_clean.loc[area_mask, 'lat'].mean()
        )
        df_clean.loc[area_mask, 'lng'] = df_clean.loc[area_mask, 'lng'].fillna(
            df_clean.loc[area_mask, 'lng'].mean()
        )
    
    # Remove any remaining rows with NaN
    df_clean = df_clean.dropna()
    
    return df_clean


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional features from existing data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with engineered features
    """
    df_eng = df.copy()
    
    # Extract city from address (last part before postal code pattern)
    df_eng['City'] = df_eng['Address'].str.extract(r'([A-Za-z\s]+),\s*ON')[0].str.strip()
    
    # Create price per location metric (area average)
    area_price_avg = df_eng.groupby('AreaName')['Price ($)'].transform('mean')
    df_eng['Area_Avg_Price'] = area_price_avg
    
    # Distance from downtown Toronto (approximate: 43.6532, -79.3832)
    downtown_lat, downtown_lng = 43.6532, -79.3832
    df_eng['Distance_From_Downtown'] = np.sqrt(
        (df_eng['lat'] - downtown_lat)**2 + (df_eng['lng'] - downtown_lng)**2
    )
    
    # Area property count (market size indicator)
    area_counts = df_eng.groupby('AreaName').size()
    df_eng['Area_Property_Count'] = df_eng['AreaName'].map(area_counts)
    
    return df_eng


def prepare_features(
    df: pd.DataFrame,
    fit_encoders: bool = True,
    label_encoder_area: Optional[LabelEncoder] = None,
    label_encoder_city: Optional[LabelEncoder] = None,
    scaler: Optional[StandardScaler] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Prepare features for model training/prediction by encoding categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with engineered features
    fit_encoders : bool
        If True, fit new encoders. If False, use provided encoders
    label_encoder_area : LabelEncoder, optional
        Pre-fitted encoder for AreaName
    label_encoder_city : LabelEncoder, optional
        Pre-fitted encoder for City
    scaler : StandardScaler, optional
        Pre-fitted scaler for numerical features
        
    Returns:
    --------
    Tuple[pd.DataFrame, dict]
        Processed features and dictionary of fitted encoders/scaler
    """
    df_prep = df.copy()
    encoders = {}
    
    # Encode AreaName
    if fit_encoders:
        le_area = LabelEncoder()
        df_prep['AreaName_Encoded'] = le_area.fit_transform(df_prep['AreaName'])
        encoders['area_encoder'] = le_area
    else:
        if label_encoder_area is None:
            raise ValueError("label_encoder_area must be provided when fit_encoders=False")
        # Handle unknown areas
        df_prep['AreaName_Encoded'] = df_prep['AreaName'].apply(
            lambda x: label_encoder_area.transform([x])[0] 
            if x in label_encoder_area.classes_ 
            else -1
        )
        encoders['area_encoder'] = label_encoder_area
    
    # Encode City
    if fit_encoders:
        le_city = LabelEncoder()
        df_prep['City_Encoded'] = le_city.fit_transform(df_prep['City'])
        encoders['city_encoder'] = le_city
    else:
        if label_encoder_city is None:
            raise ValueError("label_encoder_city must be provided when fit_encoders=False")
        # Handle unknown cities
        df_prep['City_Encoded'] = df_prep['City'].apply(
            lambda x: label_encoder_city.transform([x])[0] 
            if x in label_encoder_city.classes_ 
            else -1
        )
        encoders['city_encoder'] = label_encoder_city
    
    # Select features for modeling
    feature_columns = [
        'lat', 'lng', 'Distance_From_Downtown', 'Area_Avg_Price',
        'Area_Property_Count', 'AreaName_Encoded', 'City_Encoded'
    ]
    
    X = df_prep[feature_columns]
    
    # Optional scaling
    if scaler is not None or fit_encoders:
        if fit_encoders:
            scaler_obj = StandardScaler()
            X_scaled = scaler_obj.fit_transform(X)
            encoders['scaler'] = scaler_obj
        else:
            X_scaled = scaler.transform(X)
            encoders['scaler'] = scaler
        
        X = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
    
    return X, encoders


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable (Price)
    test_size : float
        Proportion of dataset for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_sample_data(df: pd.DataFrame, n_samples: int = 10) -> pd.DataFrame:
    """
    Get a random sample of the dataset for demonstration purposes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full dataset
    n_samples : int
        Number of samples to return
        
    Returns:
    --------
    pd.DataFrame
        Sample subset of the dataset
    """
    return df.sample(n=min(n_samples, len(df)), random_state=42)


def save_encoders(encoders: dict, directory: str = "models") -> None:
    """
    Save fitted encoders and scalers to disk.
    
    Parameters:
    -----------
    encoders : dict
        Dictionary containing fitted encoders/scaler
    directory : str
        Directory to save encoder files
    """
    for name, encoder in encoders.items():
        filepath = f"{directory}/{name}.joblib"
        joblib.dump(encoder, filepath)


def load_encoders(directory: str = "models") -> dict:
    """
    Load fitted encoders and scalers from disk.
    
    Parameters:
    -----------
    directory : str
        Directory containing encoder files
        
    Returns:
    --------
    dict
        Dictionary of loaded encoders/scaler
    """
    encoders = {}
    encoder_names = ['area_encoder', 'city_encoder', 'scaler']
    
    for name in encoder_names:
        filepath = f"{directory}/{name}.joblib"
        try:
            encoders[name] = joblib.load(filepath)
        except FileNotFoundError:
            # Scaler is optional
            if name != 'scaler':
                raise FileNotFoundError(f"Encoder {name} not found at {filepath}")
    
    return encoders
