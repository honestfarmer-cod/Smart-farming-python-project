# Smart Farming Python Project

This project analyzes smart farming trial results and recommends the best crop varieties for each soil type.

## Datasets Used
The program reads processed datasets from:
- data/processed/trial_results_summary_by_soilvariety.csv
- data/processed/soil_types_cleaned.csv
- data/processed/crop_varieties_standardized.csv

Note: If the processed summary dataset is detected (no `soil_type_id` column), the program automatically skips merging and uses the processed file directly.

---

## How to Run (Windows PowerShell)

```powershell
# 1) Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install requirements
pip install -r requirements.txt

# 3) Run the program
python project.py

# 4) Run tests
python -m pytest
```

---

## Class: CropYieldPredictor

A machine learning wrapper for yield prediction.

### Key Methods:

| Method | Purpose | Returns |
|------|---------|---------|
| `fit(X_train, y_train)` | Train the model | `dict` with `r2_score`, `rmse` |
| `predict(X_test)` | Predict yields | NumPy array of predictions |
| `get_feature_importance(top_n)` | Feature importance ranking | DataFrame |

### Attributes:
- `model`: RandomForestRegressor (100 trees, `random_state=42`)
- `scaler`: StandardScaler for feature normalization
- `label_encoders`: Dictionary of LabelEncoder objects for categorical features
- `is_trained`: Boolean flag for model state

---

## Core Functions

### 1) load_and_prepare_data(trial_file, soil_file)
**Purpose:** Load and merge trial + soil data

**Behavior:**
- Loads CSV files if they exist
- Auto-generates sample data if files are missing
- Merges on `soil_type_id` (when applicable)
- Handles missing values (drops rows with NaN in key columns)
- Generates synthetic yield values for testing (if missing)

**Returns:** DataFrame with merged, cleaned data

**Sample columns:**
```text
trial_id, field_id, variety_name, soil_name, ph,
organic_matter_percent, nitrogen_mg_kg, yield_kg_ha
```

---

### 2) `preprocess_features(data)`
**Purpose:** Transform raw data into ML-ready features.

**Transformations:**
- Encode categorical variables (`variety_name`, `soil_name`)
- Select numeric features (nitrogen, organic matter, pH)
- Fill missing values with column means

**Returns:**
`(X_features, y_target, feature_names, label_encoders)`

---

### 3) `generate_recommendations(data, model, scaler_data)`
**Purpose:** Generate variety recommendations for each soil type.

**Logic:**
For each soil type:
- Test all crop varieties
- Predict yield for each combination
- Rank by predicted performance
- Return top recommendation

**Returns:** DataFrame with recommendations

---

### 4) `save_results(predictions, recommendations, output_dir)`
**Purpose:** Export analysis results to CSV files.

**Creates:**
- `outputs/yield_predictions.csv` - Full prediction dataset
- `outputs/variety_recommendations.csv` - Recommended varieties

**Returns:** `(predictions_file, recommendations_file)`

---

### 5) `main()`
**Purpose:** Orchestrate the entire workflow.

**Workflow Steps:**
- Load and prepare data
- Preprocess features
- Split into train/test (80/20 split)
- Train CropYieldPredictor model
- Evaluate on test set
- Generate predictions for all data
- Generate variety recommendations
- Analyze feature importance
- Save results to CSV

## Test Coverage (8 test functions + 1 integration test)

### 1. test_predictor_fit_returns_valid_metrics()
Tests: CropYieldPredictor.fit() method

Corner cases:

Returns dict with r2_score, rmse keys

R² score in valid range [0, 1]

RMSE is non-negative

---

### 2. test_predictor_fit_with_larger_dataset()
Tests: Scalability with larger data

Corner cases:

Works with 100+ records

Handles mixed feature types

Metrics are consistent

---

### 3. test_predictor_predict_requires_training()
Tests: Error handling - predict before training

Corner cases:

Raises ValueError if untrained

Error message is informative

---

### 4. test_predictor_predict_returns_non_negative_yields()
Tests: CropYieldPredictor.predict() correctness

Corner cases:

All predictions >= 0 (physical validity)

Returns numpy array

Array length matches input

---

### 5. test_load_and_prepare_data_creates_sample_data()
Tests: Automatic data generation

Corner cases:

Creates sample if files missing

Returns DataFrame with required columns

No missing values in critical columns

---

### 6. test_load_and_prepare_data_handles_missing_values()
Tests: Data cleaning functionality

Corner cases:

Removes rows with NaN

No NaN in key columns after cleaning

Retains sufficient data

---

### 7. test_preprocess_features_returns_valid_outputs()
Tests: Feature preprocessing

Corner cases:

Returns 4-tuple (X, y, names, encoders)

X has only numeric values

X and y same length

No NaN values

---

### 8. test_preprocess_features_encodes_varieties()
Tests: Categorical encoding

Corner cases:

Variety names encoded to integers

Encoders persist in dict

Encoded values in reasonable range.

---
