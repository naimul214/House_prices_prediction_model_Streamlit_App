"""
Unit tests for the predict module.

Tests prediction functions, input validation, and feature preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.predict import validate_input, format_prediction_output, get_feature_names
from src.utils import validate_csv_upload, format_currency, validate_price


def test_validate_input_valid():
    """Test input validation with valid data."""
    address = "123 Main St Toronto, ON"
    area_name = "Downtown"
    lat = 43.65
    lng = -79.38
    
    is_valid, error_msg = validate_input(address, area_name, lat, lng)
    
    assert is_valid == True
    assert error_msg == ""


def test_validate_input_invalid_latitude():
    """Test input validation with invalid latitude."""
    address = "123 Main St Toronto, ON"
    area_name = "Downtown"
    lat = 100.0  # Invalid: outside Ontario range
    lng = -79.38
    
    is_valid, error_msg = validate_input(address, area_name, lat, lng)
    
    assert is_valid == False
    assert "Latitude" in error_msg


def test_validate_input_invalid_longitude():
    """Test input validation with invalid longitude."""
    address = "123 Main St Toronto, ON"
    area_name = "Downtown"
    lat = 43.65
    lng = 0.0  # Invalid: outside Ontario range
    
    is_valid, error_msg = validate_input(address, area_name, lat, lng)
    
    assert is_valid == False
    assert "Longitude" in error_msg


def test_validate_input_empty_address():
    """Test input validation with empty address."""
    address = ""
    area_name = "Downtown"
    lat = 43.65
    lng = -79.38
    
    is_valid, error_msg = validate_input(address, area_name, lat, lng)
    
    assert is_valid == False
    assert "Address" in error_msg


def test_format_prediction_output():
    """Test prediction output formatting."""
    predictions = np.array([500000, 750000, 1000000])
    lower_bounds = np.array([450000, 700000, 950000])
    upper_bounds = np.array([550000, 800000, 1050000])
    
    results = format_prediction_output(predictions, lower_bounds, upper_bounds)
    
    assert len(results) == 3
    assert results[0]['predicted_price'] == 500000
    assert '$' in results[0]['formatted_price']
    assert 'confidence_interval' in results[0]


def test_get_feature_names():
    """Test feature names retrieval."""
    features = get_feature_names()
    
    assert isinstance(features, list)
    assert len(features) > 0
    assert 'lat' in features
    assert 'lng' in features


def test_validate_csv_upload_valid():
    """Test CSV upload validation with valid data."""
    df = pd.DataFrame({
        'Address': ['123 Main St'],
        'AreaName': ['Downtown'],
        'lat': [43.65],
        'lng': [-79.38]
    })
    
    is_valid, error_msg, missing = validate_csv_upload(df)
    
    assert is_valid == True
    assert error_msg == ""
    assert len(missing) == 0


def test_validate_csv_upload_missing_columns():
    """Test CSV upload validation with missing columns."""
    df = pd.DataFrame({
        'Address': ['123 Main St'],
        'lat': [43.65]
    })
    
    is_valid, error_msg, missing = validate_csv_upload(df)
    
    assert is_valid == False
    assert "Missing required columns" in error_msg
    assert 'AreaName' in missing
    assert 'lng' in missing


def test_format_currency():
    """Test currency formatting."""
    assert format_currency(500000) == "$500,000.00"
    assert format_currency(1234567.89) == "$1,234,567.89"
    assert format_currency(0) == "$0.00"


def test_validate_price():
    """Test price validation."""
    # Valid prices
    is_valid, _ = validate_price(500000)
    assert is_valid == True
    
    is_valid, _ = validate_price(1000000)
    assert is_valid == True
    
    # Invalid prices
    is_valid, error_msg = validate_price(-100)
    assert is_valid == False
    assert "positive" in error_msg
    
    is_valid, error_msg = validate_price(5000)
    assert is_valid == False
    assert "too low" in error_msg
    
    is_valid, error_msg = validate_price(100000000)
    assert is_valid == False
    assert "too high" in error_msg


def test_prediction_range():
    """Test that predictions fall within reasonable ranges."""
    predictions = np.array([500000, 750000, 1000000])
    
    # All predictions should be positive
    assert np.all(predictions > 0)
    
    # Predictions should be reasonable (between $10k and $50M)
    assert np.all(predictions >= 10000)
    assert np.all(predictions <= 50000000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
