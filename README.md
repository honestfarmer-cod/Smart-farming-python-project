# Smart Farming Crop Yield Prediction & Recommendation System

This repository contains our **Python Final Project (2025/2026)** for the *Masters in Data Science applied to agricultural and food sciences, environment, and forestry engineering*.

The project builds a complete workflow to:
- load smart-farming field trial + soil datasets (real files or demo mode)
- train a machine learning model to **predict crop yield (kg/ha)**
- generate **best crop variety recommendations per soil texture class**
- export results into CSV files for reporting

---

## What the Program Does

When you run `project.py`, it performs these steps:

1. **Load & prepare data**  
   Reads the raw trial dataset and merges soil composition information.

2. **Preprocess features**  
   Encodes categorical columns and selects numeric features for ML.

3. **Train/Test split (80/20)**  
   Splits data into training and testing sets.

4. **Train ML model**  
   Uses a `RandomForestRegressor` to learn yield patterns.

5. **Evaluate model**
   Prints **R²** and **RMSE** performance on the test set.

6. **Predict yields**
   Predicts yield for all available trial records.

7. **Generate variety recommendations**
   For each soil type, finds the variety with the best predicted yield.

8. **Save outputs**
   Saves predictions and recommendations into `outputs/`.

---

## Project Structure

```
smart-farming-python-project/
│
├── project.py
├── test_project.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── trials_raw.csv
│   │   ├── yield_raw.csv        (optional)
│   │   └── soil_raw.csv         (optional)
│   │
│   └── processed/
│       ├── soil_types_cleaned.csv
│       ├── crop_varieties_standardized.csv
│       └── trial_results_summary_by_soilvariety.csv (reference)
│
├── outputs/
│   ├── yield_predictions.csv
│   └── variety_recommendations.csv
│
└── docs/
```

---

## Datasets Used

The program is designed to work with the datasets from our earlier **DMS project**.

### Required file (recommended)
Place inside: `data/raw/`
- `trials_raw.csv`

### Soil data (required for merge)
Place inside: `data/processed/`
- `soil_types_cleaned.csv`

### Optional files
Place inside: `data/raw/`
- `yield_raw.csv`  
  Used only if it includes `trial_code` and `yield_kg_ha` (otherwise yield is read from `trials_raw.csv`)

- `soil_raw.csv`  
  Not required if you already use the cleaned soil dataset.

### Demo / test mode
If real CSV files are missing, the program automatically generates **sample demo data** so it can still run.

---

## Installation

### 1) Create and activate a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install requirements

```powershell
pip install -r requirements.txt
```

---

## How to Run

From the project root directory:

```bash
python project.py
```

---

## Output Files

After a successful run, two files will be created inside `outputs/`:

- `outputs/yield_predictions.csv`  
  Contains the original trial data + predicted yield and absolute prediction error.

- `outputs/variety_recommendations.csv`  
  Shows the recommended variety for each soil texture class and its predicted yield.

---

## Class: `CropYieldPredictor`

`CropYieldPredictor` is a wrapper around a machine learning model that predicts crop yield.

### Key Methods

| Method | Purpose | Returns |
|------|---------|---------|
| `fit(X_train, y_train)` | Train the model | `dict` with `r2_score`, `rmse` |
| `predict(X_test)` | Predict yields | NumPy array of predictions |
| `get_feature_importance(top_n)` | Show most important features | DataFrame |

### Key Attributes

- `model`: `RandomForestRegressor(n_estimators=100, random_state=42)`
- `scaler`: `StandardScaler()` for feature scaling
- `label_encoders`: encoders for categorical columns
- `feature_names`: list of feature columns used in training
- `is_trained`: indicates whether training has been completed

---

## Core Functions (project.py)

### 1) `load_and_prepare_data(trials_file, yield_file, soil_file)`
Loads trial and soil data, and merges them into one dataset.

**Behavior:**
- reads CSV files if they exist
- optionally merges yield values using `trial_code`
- standardizes merge keys (case/spacing)
- merges soil texture class with soil composition information
- generates demo sample data if files are missing

---

### 2) `preprocess_features(data)`
Prepares ML-ready features and target values.

**Transformations:**
- encodes categorical columns:
  - `crop`
  - `variety_name`
  - `soil_texture_class`
- selects numeric + encoded features for training
- fills missing values using mean imputation

**Returns:**
`X, y, feature_cols, label_encoders`

---

### 3) `generate_recommendations(data, model)`
Predicts yields for each variety and recommends the best variety per soil type.

**Returns:**  
A DataFrame with columns:
- `soil_texture_class`
- `recommended_variety`
- `predicted_yield_kg_ha`

---

### 4) `save_results(predictions, recommendations, output_dir="outputs")`
Saves:
- `yield_predictions.csv`
- `variety_recommendations.csv`

---

### 5) `main()`
Runs the full workflow end-to-end and prints a summary report.

---

## How to Run Tests

Run all tests:

```bash
pytest -v
```

Or run only this file:

```bash
pytest test_project.py -v
```

### Test Coverage Summary
The test suite validates:
- training metrics are returned correctly
- prediction errors are handled correctly (must train before predict)
- predictions are non-negative
- data loading works even when files are missing (demo mode)
- preprocessing produces numeric features without NaNs
- an end-to-end integration test passes

---

## Notes / Assumptions
- Yields are predicted in **kg/ha**
- Negative predicted yields are clipped to 0 (physically meaningful)
- The project is designed to be runnable even without real datasets (demo mode)

---

## Authors
- **Aster Noel Dsouza** (Student ID: **29211**)
- **David Heleno Bebiano Da Costa Morais** (Student ID: **29400**)
