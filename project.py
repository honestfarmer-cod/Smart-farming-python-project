import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

#A machine learning model for predicting crop yields based on soil,
    
class CropYieldPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False

#Train the yield prediction model.
    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True
        y_pred = self.model.predict(X_scaled) 
        r2 = r2_score(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        return {'r2_score': r2, 'rmse': rmse}
 
    #Predict crop yield for new data.
    def predict(self, X_test):
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_scaled = self.scaler.transform(X_test)
        predictions = self.model.predict(X_scaled)
        return np.maximum(predictions, 0)  # Ensure non-negative predictions
 
    #Get the most important features for yield prediction.
    def get_feature_importance(self, top_n=5):
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
 
#Loading the simulated trial data and soil data, merge them, and prepare them for analysis.(copied from the DMS Project)
   
def load_and_prepare_data(trials_file, yield_file, soil_file):
    """
    Loads the trials dataset and optionally merges soil info.
    If files do not exist, generates demo data for testing.
    """

    # ---- REAL DATA MODE ----
    if os.path.exists(trials_file) and os.path.exists(soil_file):
        trials = pd.read_csv(trials_file)
        soils = pd.read_csv(soil_file)

        # yield_file is optional because trials_raw.csv already contains yield_kg_ha
        if os.path.exists(yield_file):
            yield_df = pd.read_csv(yield_file)

            # Merge only if yield_df contains trial_code + yield_kg_ha
            if "trial_code" in yield_df.columns and "yield_kg_ha" in yield_df.columns:
                trials = trials.merge(
                    yield_df[["trial_code", "yield_kg_ha"]],
                    on="trial_code",
                    how="left",
                    suffixes=("", "_from_yieldfile")
                )

    # ---- DEMO DATA MODE (for tests) ----
    else:
        trials = pd.DataFrame({
            "trial_code": [f"T{i}" for i in range(1, 121)],
            "location_name": np.random.choice(["Centro", "Norte"], 120),
            "year": np.random.choice([2021, 2022, 2023, 2024], 120),
            "crop": np.random.choice(["Wheat", "Rice", "Maize", "Barley"], 120),
            "variety_name": np.random.choice(["Wheat_V1", "Wheat_V2", "Rice_V1", "Rice_V2"], 120),
            "soil_texture_class": np.random.choice(["Clay Loam", "Sandy Loam", "Loam", "Clay"], 120),
            "area_ha": np.random.uniform(2.0, 3.0, 120).round(1),
            "seed_rate_kg_ha": np.random.uniform(170, 210, 120).round(0),
            "n_kg_ha": np.random.uniform(90, 120, 120).round(0),
            "p_kg_ha": np.random.uniform(55, 75, 120).round(0),
            "k_kg_ha": np.random.uniform(50, 75, 120).round(0),
            "irrigation_mm": np.random.uniform(380, 820, 120).round(0),
        })

        # Generate realistic yield
        trials["yield_kg_ha"] = (
            trials["n_kg_ha"] * 10
            + trials["p_kg_ha"] * 5
            + trials["k_kg_ha"] * 4
            + trials["irrigation_mm"] * 2
            + np.random.normal(0, 200, len(trials))
        ).clip(lower=500)

        soils = pd.DataFrame({
            "soil_name": ["Clay Loam", "Sandy Loam", "Loam", "Clay"],
            "sand_percentage": [30, 60, 40, 20],
            "silt_percentage": [40, 25, 40, 30],
            "clay_percentage": [30, 15, 20, 50],
            "pH_range_min": [6.5, 6.8, 6.7, 6.2],
            "pH_range_max": [7.2, 7.6, 7.4, 7.0],
        })

    # ✅ Standardize merge keys to avoid mismatches (case/spacing issues)
    trials["soil_texture_class"] = trials["soil_texture_class"].astype(str).str.strip().str.lower()
    soils["soil_name"] = soils["soil_name"].astype(str).str.strip().str.lower()

    # ---- MERGE TRIALS + SOIL INFO ----
    merged_data = trials.merge(
        soils,
        left_on="soil_texture_class",
        right_on="soil_name",
        how="left"
    )

    return merged_data


# Prepare features for machine learning model
def preprocess_features(data):
    df = data.copy()

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ["crop", "variety_name", "soil_texture_class"]

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Feature columns from trials_raw.csv (REAL column names)
    feature_cols = [
        "seed_rate_kg_ha",
        "n_kg_ha",
        "p_kg_ha",
        "k_kg_ha",
        "irrigation_mm",
        "area_ha",
        "crop_encoded",
        "variety_name_encoded",
        "soil_texture_class_encoded"
    ]

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean(numeric_only=True))

    if "yield_kg_ha" not in df.columns:
        raise KeyError("yield_kg_ha not found. Check trials_raw.csv")
    y = df["yield_kg_ha"]

    return X, y, feature_cols, label_encoders

#Generate crop variety recommendations based on predicted yields.

def generate_recommendations(data, model):
    _, _, _, encoders = preprocess_features(data)

    soil_types = data["soil_texture_class"].unique()
    varieties = data["variety_name"].unique()

    recommendations = []

    for soil in soil_types:
        soil_data = data[data["soil_texture_class"] == soil].iloc[0]

        best_variety = None
        best_yield = -1

        for variety in varieties:
            features = pd.DataFrame({
                "seed_rate_kg_ha": [soil_data["seed_rate_kg_ha"]],
                "n_kg_ha": [soil_data["n_kg_ha"]],
                "p_kg_ha": [soil_data["p_kg_ha"]],
                "k_kg_ha": [soil_data["k_kg_ha"]],
                "irrigation_mm": [soil_data["irrigation_mm"]],
                "area_ha": [soil_data["area_ha"]],
                "crop_encoded": [encoders["crop"].transform([soil_data["crop"]])[0]],
                "variety_name_encoded": [encoders["variety_name"].transform([variety])[0]],
                "soil_texture_class_encoded": [encoders["soil_texture_class"].transform([soil])[0]],
            })

            predicted_yield = model.predict(features)[0]

            if predicted_yield > best_yield:
                best_yield = predicted_yield
                best_variety = variety

        recommendations.append({
            "soil_texture_class": soil,
            "recommended_variety": best_variety,
            "predicted_yield_kg_ha": round(best_yield, 2)
        })

    return pd.DataFrame(recommendations)

#Save prediction and recommendation results to CSV files.
def save_results(predictions, recommendations, output_dir='outputs'):
# Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
# Save prediction results
    pred_file = os.path.join(output_dir, 'yield_predictions.csv')
    predictions.to_csv(pred_file, index=False)
    print(f"✓ Predictions saved to {pred_file}")
    
# Save recommendations
    rec_file = os.path.join(output_dir, 'variety_recommendations.csv')
    recommendations.to_csv(rec_file, index=False)
    print(f"✓ Recommendations saved to {rec_file}")
    
    return pred_file, rec_file

def main():
    
#Main function orchestrating the entire crop yield prediction workflow.
    
  
    print("="*70)
    print("CROP YIELD PREDICTION & RECOMMENDATION SYSTEM")
    print("="*70)
    
# Step 1: Load and prepare data
    print("\n[Step 1] Loading and preparing data...")
    data = load_and_prepare_data(
    "data/raw/trials_raw.csv",
    "data/raw/yield_raw.csv",
    "data/processed/soil_types_cleaned.csv"
)
    print(f"  ✓ Loaded {len(data)} trial records")
    print(f"  ✓ Data columns: {list(data.columns)[:5]}...")
    
# Step 2: Preprocess features
    print("\n[Step 2] Preprocessing features...")
    X, y, feature_names, encoders = preprocess_features(data)
    print(f"  ✓ Features: {feature_names}")
    print(f"  ✓ Target range: {y.min():.2f} - {y.max():.2f} kg/ha")
    
# Step 3: Split data
    print("\n[Step 3] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  ✓ Training set: {len(X_train)} records")
    print(f"  ✓ Test set: {len(X_test)} records")
    
# Step 4: Train model
    print("\n[Step 4] Training CropYieldPredictor model...")
    predictor = CropYieldPredictor()
    predictor.feature_names = feature_names
    metrics = predictor.fit(X_train, y_train)
    print(f"  ✓ Training R² score: {metrics['r2_score']:.4f}")
    print(f"  ✓ Training RMSE: {metrics['rmse']:.2f} kg/ha")
    
# Evaluate on test set
    y_pred_test = predictor.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"  ✓ Test R² score: {test_r2:.4f}")
    print(f"  ✓ Test RMSE: {test_rmse:.2f} kg/ha")
    
# Step 5: Generate predictions for all data
    print("\n[Step 5] Generating predictions for all records...")
    all_predictions = predictor.predict(X)
    data['predicted_yield_kg_ha'] = all_predictions
    data['prediction_error'] = abs(data['yield_kg_ha'] - data['predicted_yield_kg_ha'])
    print(f"  ✓ Predicted yields for {len(data)} records")
    print(f"  ✓ Mean prediction error: {data['prediction_error'].mean():.2f} kg/ha")
    
# Step 6: Generate recommendations
    print("\n[Step 6] Generating variety recommendations...")
    recommendations = generate_recommendations(data, predictor)
    print(f"  ✓ Generated {len(recommendations)} recommendations")
    print(f"  ✓ Average recommended yield: {recommendations['predicted_yield_kg_ha'].mean():.2f} kg/ha")
    
# Display feature importance
    print("\n[Step 7] Feature importance analysis...")
    importance = predictor.get_feature_importance(top_n=5)
    for idx, row in importance.iterrows():
        print(f"  • {row['feature']}: {row['importance']:.4f}")
    
# Step 8: Save results
    print("\n[Step 8] Saving results to CSV files...")
    pred_file, rec_file = save_results(data, recommendations)
    
# Summary statistics
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY STATISTICS")
    print("="*70)
    summary = {
        'total_records': len(data),
        'avg_actual_yield': data['yield_kg_ha'].mean(),
        'avg_predicted_yield': data['predicted_yield_kg_ha'].mean(),
        'model_r2_test': test_r2,
        'model_rmse_test': test_rmse,
        'recommendations_count': len(recommendations),
        'top_feature': importance.iloc[0]['feature']
    }
    
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("="*70)
    
    return summary


if __name__ == "__main__":
    main()