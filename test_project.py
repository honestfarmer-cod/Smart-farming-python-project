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
 

 #Test that load_and_prepare_data() creates sample data if files don't exist.
def test_load_and_prepare_data_creates_sample_data():
# Execute: Call with non-existent files to trigger sample generation
    data = load_and_prepare_data('nonexistent_trials.csv', 'nonexistent_yield.csv', 'nonexistent_soil.csv')
# Assert
    assert isinstance(data, pd.DataFrame), "Should return DataFrame"
    assert len(data) > 0, "Should have data rows"
    
    required_cols = ['yield_kg_ha', 'ph', 'organic_matter_percent', 'nitrogen_mg_kg']
    for col in required_cols:
        assert col in data.columns, f"Should contain '{col}' column"
        assert data[col].notna().all(), f"'{col}' should have no missing values"
    
    assert (data['yield_kg_ha'] >= 0).all(), "Yields should be non-negative"
    assert (data['ph'] >= 6.5).all() and (data['ph'] <= 7.5).all(), "pH should be in valid range"



#Test that load_and_prepare_data() properly handles missing data.
def test_load_and_prepare_data_handles_missing_values():
 # Execute
    data = load_and_prepare_data('nonexistent_trials.csv', 'nonexistent_yield.csv', 'nonexistent_soil.csv')
# Assert
    critical_cols = ['ph', 'organic_matter_percent', 'nitrogen_mg_kg', 'yield_kg_ha']
    for col in critical_cols:
        assert data[col].isna().sum() == 0, f"'{col}' should have no missing values after prepare"
    
    assert len(data) >= 50, "Should retain sufficient data after cleaning"   


#Test that preprocess_features() returns correctly structured outputs.
def test_preprocess_features_returns_valid_outputs():
# Setup: Create sample data
    data = load_and_prepare_data('nonexistent_trials.csv', 'nonexistent_yield.csv', 'nonexistent_soil.csv')
# Execute
    X, y, feature_names, encoders = preprocess_features(data)
# Assert
    assert isinstance(X, pd.DataFrame), "X should be DataFrame"
    assert isinstance(y, pd.Series), "y should be Series"
    assert isinstance(feature_names, list), "feature_names should be list"
    assert isinstance(encoders, dict), "encoders should be dict"
    
    assert len(X) == len(y), "X and y should have same length"
    assert len(feature_names) >= 3, "Should have at least 3 features"
    assert len(encoders) >= 1, "Should have at least 1 encoder for categorical vars"
# Check X contains only numeric values
    assert X.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all(), \
        "All features should be numeric"
# Check no Na values
    assert X.isna().sum().sum() == 0, "X should have no missing values"
    assert y.isna().sum() == 0, "y should have no missing values"


#Test that preprocess_features() properly encodes categorical variety data.
def test_preprocess_features_encodes_varieties():
# Setup
    data = load_and_prepare_data('nonexistent_trials.csv', 'nonexistent_yield.csv', 'nonexistent_soil.csv')
# Execute
    X, y, feature_names, encoders = preprocess_features(data)
# Assert
    assert 'variety_name_encoded' in feature_names, "Should have variety encoding"
    assert 'variety_name' in encoders, "Should have variety_name encoder"
# Check that encoded values are integers in reasonable range
    variety_encoded = X['variety_name_encoded']
    assert variety_encoded.dtype in [np.int32, np.int64], "Should be encoded as integers"
    assert variety_encoded.min() >= 0, "Encoded values should be non-negative"

# Incoperate integration test
#Integration test: Load -> Preprocess -> Train -> Predict full workflow.
def test_full_pipeline_integration():
# Step 1: Load data
    data = load_and_prepare_data('nonexistent_trials.csv', 'nonexistent_yield.csv', 'nonexistent_soil.csv')
    assert len(data) > 0, "Data loading failed"
    
# Step 2: Preprocess
    X, y, feature_names, encoders = preprocess_features(data)
    assert len(X) > 0 and len(y) > 0, "Preprocessing failed"
    
# Step 3: Train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    predictor = CropYieldPredictor()
    predictor.feature_names = feature_names
    metrics = predictor.fit(X_train, y_train)
    assert metrics['r2_score'] >= 0, "Training failed"
    
# Step 4: Predict
    predictions = predictor.predict(X_test)
    assert len(predictions) == len(X_test), "Prediction count mismatch"
    assert all(pred >= 0 for pred in predictions), "Invalid predictions"
    
# Step 5: Validate
    assert len(data) >= len(X), "Data integrity check failed"
    print(f"\n✓ Integration test passed: {len(data)} records processed, "
          f"R²={metrics['r2_score']:.4f}, RMSE={metrics['rmse']:.2f} kg/ha")


if __name__ == "__main__":
# Run tests with: python -m pytest test_project.py -v
    pytest.main([__file__, "-v"])
