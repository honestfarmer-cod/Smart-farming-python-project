import pytest
import pandas as pd
import numpy as np
from project import (
    CropYieldPredictor,
    load_and_prepare_data,
    preprocess_features,
    generate_recommendations
)
#Test that CropYieldPredictor.fit() returns valid metrics dictionary.
def test_predictor_fit_returns_valid_metrics():
# Setup: Create sample data
    X_train = pd.DataFrame({
        'nitrogen_mg_kg': [40, 50, 60, 45, 55],
        'organic_matter_percent': [2.5, 3.0, 3.5, 2.8, 3.2],
        'ph': [6.8, 7.0, 6.9, 7.1, 6.7],
        'variety_name_encoded': [0, 1, 2, 0, 1],
        'soil_name_encoded': [0, 1, 2, 1, 0]
    })
    y_train = pd.Series([2000, 2500, 2800, 2200, 2600])
    
# Execute
    predictor = CropYieldPredictor()
    metrics = predictor.fit(X_train, y_train)
    
# Assert
    assert isinstance(metrics, dict), "fit() should return a dictionary"
    assert 'r2_score' in metrics, "Metrics should include 'r2_score'"
    assert 'rmse' in metrics, "Metrics should include 'rmse'"
    assert 0 <= metrics['r2_score'] <= 1, "R² score must be between 0 and 1"
    assert metrics['rmse'] >= 0, "RMSE must be non-negative"
    assert predictor.is_trained == True, "Model should be marked as trained"

#Test that CropYieldPredictor.fit() works with larger, more realistic data.
    
def test_predictor_fit_with_larger_dataset():
# Setup: Create larger realistic dataset
    np.random.seed(42)
    X_train = pd.DataFrame({
        'nitrogen_mg_kg': np.random.uniform(20, 70, 100),
        'organic_matter_percent': np.random.uniform(1.5, 5.0, 100),
        'ph': np.random.uniform(6.5, 7.5, 100),
        'variety_name_encoded': np.random.randint(0, 8, 100),
        'soil_name_encoded': np.random.randint(0, 8, 100)
    })
    y_train = pd.Series(np.random.uniform(1500, 3500, 100))
    
# Execute
    predictor = CropYieldPredictor()
    metrics = predictor.fit(X_train, y_train)
    
# Assert
    assert metrics['rmse'] > 0, "RMSE should be positive on real data"
    assert metrics['r2_score'] >= 0, "R² should be non-negative"




#Test that CropYieldPredictor.predict() raises error if called before training.
def test_predictor_predict_requires_training():
    
    
 
# Setup
    predictor = CropYieldPredictor()
    X_test = pd.DataFrame({
        'nitrogen_mg_kg': [50],
        'organic_matter_percent': [3.0],
        'ph': [7.0],
        'variety_name_encoded': [0],
        'soil_name_encoded': [0]
    })
    
# Execute & Assert
    with pytest.raises(ValueError) as exc_info:
        predictor.predict(X_test)
    
    assert "trained" in str(exc_info.value).lower(), "Error should mention training"


#Test that CropYieldPredictor.predict() returns only non-negative yield values.
def test_predictor_predict_returns_non_negative_yields():
# Setup: Train model
    X_train = pd.DataFrame({
        'nitrogen_mg_kg': [40, 50, 60, 45, 55, 48, 52, 42],
        'organic_matter_percent': [2.5, 3.0, 3.5, 2.8, 3.2, 2.9, 3.1, 2.6],
        'ph': [6.8, 7.0, 6.9, 7.1, 6.7, 6.95, 7.05, 6.75],
        'variety_name_encoded': [0, 1, 2, 0, 1, 2, 0, 1],
        'soil_name_encoded': [0, 1, 2, 1, 0, 2, 1, 0]
    })
    y_train = pd.Series([2000, 2500, 2800, 2200, 2600, 2400, 2550, 2100])
    
    predictor = CropYieldPredictor()
    predictor.fit(X_train, y_train)
    
# Test: Predict on multiple samples
    X_test = pd.DataFrame({
        'nitrogen_mg_kg': [45, 55, 50],
        'organic_matter_percent': [2.8, 3.2, 3.0],
        'ph': [7.0, 6.9, 7.1],
        'variety_name_encoded': [1, 0, 2],
        'soil_name_encoded': [1, 0, 2]
    })
    
# Execute
    predictions = predictor.predict(X_test)
    
# Assert
    assert isinstance(predictions, np.ndarray), "Should return numpy array"
    assert len(predictions) == 3, "Should have 3 predictions for 3 samples"
    assert all(pred >= 0 for pred in predictions), "All predictions should be non-negative"

    
