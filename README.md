# Smart Farming Python Project

This project analyzes smart farming trial results and recommends the best crop varieties for each soil type.

## Datasets Used
The program reads processed datasets from:
- data/processed/trial_results_summary_by_soilvariety.csv
- data/processed/soil_types_cleaned.csv
- data/processed/crop_varieties_standardized.csv

## How to Run
### 1) Create and activate a virtual environment (recommended)

Windows PowerShell:
python -m venv .venv
.venv\Scripts\Activate.ps1

### 2) Install requirements
pip install -r requirements.txt

### 3) Run the program
python project.py

## Tests
Run tests with:
pytest

Class: CropYieldPredictor
A machine learning wrapper for yield prediction.

Key Methods:

Method	Purpose	Returns
fit(X_train, y_train)	Train the model	dict with r2_score, rmse
predict(X_test)	Predict yields	numpy array of predictions
get_feature_importance(top_n)	Feature importance ranking	DataFrame
Attributes:

model: RandomForestRegressor (100 trees, random_state=42)

scaler: StandardScaler for feature normalization

label_encoders: Dict of LabelEncoders for categorical features

is_trained: Boolean flag for model state

Core Functions
1. load_and_prepare_data(trial_file, soil_file)
Purpose: Load and merge trial + soil data

Behavior:

Loads CSV files if they exist

Auto-generates realistic sample data if files missing

Merges on soil_type_id

Handles missing values (drops rows with NaN in key columns)

Generates synthetic yield values for testing

Returns: DataFrame with merged, cleaned data

Sample columns:

text
trial_id, field_id, variety_name, soil_name, ph, 
organic_matter_percent, nitrogen_mg_kg, yield_kg_ha
2. preprocess_features(data)
Purpose: Transform raw data into ML-ready features

Transformations:

Encode categorical variables (variety_name, soil_name)

Select numeric features (nitrogen, organic matter, pH)

Fill missing values with column means

Create normalized feature set

Returns: (X_features, y_target, feature_names, label_encoders)

3. generate_recommendations(data, model, scaler_data)
Purpose: Generate variety recommendations for each soil type

Logic:

For each soil type:

Test all crop varieties

Predict yield for each combination

Rank by predicted performance

Return top recommendation

Returns: DataFrame with recommendations

4. save_results(predictions, recommendations, output_dir)
Purpose: Export analysis results to CSV files

Creates:

output/yield_predictions.csv - Full prediction dataset

output/variety_recommendations.csv - Recommended varieties

Returns: (predictions_file, recommendations_file)

5. main()
Purpose: Orchestrate entire workflow

Workflow Steps:

Load and prepare data

Preprocess features

Split into train/test (80/20 split)

Train CropYieldPredictor model

Evaluate on test set

Generate predictions for all data

Generate variety recommendations

Analyze feature importance

Save results to CSV