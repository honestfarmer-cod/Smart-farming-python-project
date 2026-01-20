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
    environmental, and crop variety characteristics.
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