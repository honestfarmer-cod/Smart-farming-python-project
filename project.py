import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import csv
import sys
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
            raise ValueError("No prediction after training")
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
   
def load_and_prepare_data(trial_file, soil_file):
# Try to load existing files
    if os.path.exists(trial_file) and os.path.exists(soil_file):
        trials = pd.read_csv(trial_file)
        soils = pd.read_csv(soil_file)
    else:
        # Create sample data for demonstration
        print("Creating sample data for demonstration...")
        
        # Sample trial data
        trials = pd.DataFrame({
            'trial_id': range(1, 121),
            'field_id': np.repeat(range(1, 9), 15),
            'variety_name': np.tile(['Variety A', 'Variety B', 'Variety C', 'Variety D', 
                                     'Variety E', 'Variety F', 'Variety G', 'Variety H'], 15),
            'soil_type_id': np.repeat(range(1, 9), 15),
            'planting_date': pd.date_range('2024-03-01', periods=120, freq='D'),
            'harvest_date': pd.date_range('2024-08-01', periods=120, freq='D')
        })
        
        # Sample soil data
        soils = pd.DataFrame({
            'soil_type_id': range(1, 9),
            'soil_name': ['Clay', 'Sandy Loam', 'Silt Loam', 'Clay Loam',
                         'Sandy Clay', 'Silty Clay', 'Loamy Sand', 'Peat'],
            'ph': [6.8, 7.2, 6.9, 7.1, 6.7, 6.5, 7.3, 5.9],
            'organic_matter_percent': [3.5, 2.1, 4.2, 3.8, 1.9, 2.8, 1.5, 8.2],
            'nitrogen_mg_kg': [45, 28, 52, 38, 22, 35, 18, 65]
        })
    
    # Merge datasets
    merged_data = trials.merge(soils, on='soil_type_id', how='left')
    
    # Handle missing values
    merged_data = merged_data.dropna(subset=['ph', 'organic_matter_percent', 'nitrogen_mg_kg'])
    
    # Generate realistic yield values based on soil and variety characteristics
    if 'yield_kg_ha' not in merged_data.columns:
        merged_data['yield_kg_ha'] = (
            (merged_data['nitrogen_mg_kg'] / 10) +
            (merged_data['organic_matter_percent'] * 100) +
            (merged_data['ph'] * 150) +
            np.random.normal(0, 200, len(merged_data))
        )
        merged_data['yield_kg_ha'] = merged_data['yield_kg_ha'].clip(lower=500)
    
    return merged_data

#Prepare features for machine learning model
def preprocess_features(data):
# Create a copy to avoid modifying original
    df = data.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['variety_name', 'soil_name']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Select features for the model
    feature_cols = [
        'nitrogen_mg_kg',
        'organic_matter_percent',
        'ph',
        'variety_name_encoded',
        'soil_name_encoded'
    ]
    
    # Filter to only columns that exist
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['yield_kg_ha']
    
    return X, y, feature_cols, label_encoders

#Generate crop variety recommendations based on predicted yields.

def generate_recommendations(data, model, scaler_data):
    X_features, _, feature_names, encoders = preprocess_features(data)
 # Get unique soil types and varieties
    soils = data['soil_name'].unique()
    varieties = data['variety_name'].unique()
    
    recommendations = []
    
    for soil in soils:
        soil_data = data[data['soil_name'] == soil].iloc[0]
        best_variety = None
        best_yield = 0
        results_for_soil = []
        
        for variety in varieties:
            # Create feature vector for prediction
            variety_data = data[data['variety_name'] == variety].iloc[0] if variety in data['variety_name'].values else soil_data
            
            features = pd.DataFrame({
                'nitrogen_mg_kg': [soil_data['nitrogen_mg_kg']],
                'organic_matter_percent': [soil_data['organic_matter_percent']],
                'ph': [soil_data['ph']],
                'variety_name_encoded': [encoders['variety_name'].transform([variety])[0]],
                'soil_name_encoded': [encoders['soil_name'].transform([soil])[0]]
            })
            
            predicted_yield = model.predict(features)[0]
            results_for_soil.append({
                'soil_name': soil,
                'variety_name': variety,
                'predicted_yield_kg_ha': round(predicted_yield, 2),
                'nitrogen_mg_kg': soil_data['nitrogen_mg_kg'],
                'ph': soil_data['ph'],
                'organic_matter_percent': soil_data['organic_matter_percent']
            })
            
            if predicted_yield > best_yield:
                best_yield = predicted_yield
                best_variety = variety
        
        # Add recommendation for best variety
        best_rec = next((r for r in results_for_soil if r['variety_name'] == best_variety), None)
        if best_rec:
            best_rec['rank'] = '1st - RECOMMENDED'
            recommendations.append(best_rec)
    
    return pd.DataFrame(recommendations)

#Save prediction and recommendation results to CSV files.
def save_results(predictions, recommendations, output_dir='output'):
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
    data = load_and_prepare_data('data/trial_data.csv', 'data/soil_data.csv')
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
    recommendations = generate_recommendations(data, predictor, (X, y, feature_names, encoders))
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