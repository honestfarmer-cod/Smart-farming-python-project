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
        "seed_rate_kg_ha": [180, 190, 200, 175, 195],
        "n_kg_ha": [90, 100, 110, 95, 105],
        "p_kg_ha": [60, 65, 70, 62, 68],
        "k_kg_ha": [55, 60, 65, 58, 62],
        "irrigation_mm": [500, 600, 700, 550, 650],
        "area_ha": [2.1, 2.4, 2.8, 2.2, 2.6],
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
        "seed_rate_kg_ha": np.random.uniform(170, 210, 100),
        "n_kg_ha": np.random.uniform(80, 130, 100),
        "p_kg_ha": np.random.uniform(50, 80, 100),
        "k_kg_ha": np.random.uniform(40, 80, 100),
        "irrigation_mm": np.random.uniform(350, 850, 100),
        "area_ha": np.random.uniform(2.0, 3.0, 100),
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
        "seed_rate_kg_ha": [190],
        "n_kg_ha": [100],
        "p_kg_ha": [65],
        "k_kg_ha": [60],
        "irrigation_mm": [600],
        "area_ha": [2.5],
    })
    
# Execute & Assert
    with pytest.raises(ValueError) as exc_info:
        predictor.predict(X_test)
    
    assert "trained" in str(exc_info.value).lower(), "Error should mention training"


#Test that CropYieldPredictor.predict() returns only non-negative yield values.
def test_predictor_predict_returns_non_negative_yields():
# Setup: Train model
    X_train = pd.DataFrame({
        "seed_rate_kg_ha": [180, 190, 200, 175, 195, 185, 205, 178],
        "n_kg_ha": [90, 100, 110, 95, 105, 98, 112, 92],
        "p_kg_ha": [60, 65, 70, 62, 68, 64, 72, 61],
        "k_kg_ha": [55, 60, 65, 58, 62, 59, 66, 56],
        "irrigation_mm": [500, 600, 700, 550, 650, 580, 720, 510],
        "area_ha": [2.1, 2.4, 2.8, 2.2, 2.6, 2.3, 2.9, 2.0],
    })
    y_train = pd.Series([2000, 2500, 2800, 2200, 2600, 2400, 2550, 2100])
    
    predictor = CropYieldPredictor()
    predictor.fit(X_train, y_train)
    
# Test: Predict on multiple samples
    X_test = pd.DataFrame({
        "seed_rate_kg_ha": [185, 195, 188],
        "n_kg_ha": [95, 105, 100],
        "p_kg_ha": [62, 68, 65],
        "k_kg_ha": [58, 62, 60],
        "irrigation_mm": [550, 650, 600],
        "area_ha": [2.2, 2.6, 2.4],
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
    
    required_cols = [
        "yield_kg_ha",
        "pH_range_min",
        "pH_range_max",
        "sand_percentage",
        "silt_percentage",
        "clay_percentage"    
    ]
    for col in required_cols:
        assert col in data.columns, f"Should contain '{col}' column"
        assert data[col].notna().all(), f"'{col}' should have no missing values"
    
    assert (data['yield_kg_ha'] >= 0).all(), "Yields should be non-negative"
    assert (data["pH_range_min"] >= 6.0).all() and (data["pH_range_max"] <= 8.0).all(), "pH range should be valid"



#Test that load_and_prepare_data() properly handles missing data.
def test_load_and_prepare_data_handles_missing_values():
 # Execute
    data = load_and_prepare_data('nonexistent_trials.csv', 'nonexistent_yield.csv', 'nonexistent_soil.csv')
# Assert
    critical_cols = [
        "yield_kg_ha",
        "pH_range_min",
        "pH_range_max",
        "sand_percentage",
        "silt_percentage",
        "clay_percentage"
    ]
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
