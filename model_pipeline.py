import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score # Added CV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor 
def preprocess_data(df, is_training=True):
    """Clean data and engineering features, avoiding redundancies."""
    # 0. Drop rows where 'date' is NaN (especially trailing empty rows from CSV reading)
    # 1. Sort by date and store to ensure time-based features make sense
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.dropna(subset=['date']).copy() # Drop any failed parses
    df = df.sort_values(['store_ID', 'date'])
    
    # 2. Feature Engineering: Basic Date Components
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    
    # 3. Non-Leaky Lag Features & Rolling Means
    # If we are inferencing and the validation set doesn't have sales, fill with 0 to prevent errors
    # (Though best practice is to pass historical data with the validation set)
    if 'sales' not in df.columns:
        df['sales'] = 0

    df['sales_lag_7'] = df.groupby('store_ID')['sales'].shift(7).fillna(0)
    df['sales_lag_14'] = df.groupby('store_ID')['sales'].shift(14).fillna(0)
    
    # Safely calculate rolling mean to avoid ValueError on groups with 1 row
    def safe_rolling_mean(x):
        if len(x) < 2:
            return pd.Series(0.0, index=x.index)
        # Shift first, then roll
        return x.shift(1).rolling(window=7, min_periods=1).mean()
        
    df['rolling_mean_7'] = df.groupby('store_ID')['sales'].transform(safe_rolling_mean).fillna(0)
    
    # 5. Row filtering: Only relevant for training
    if is_training:
        df = df[(df['open'] != 0) & (df['sales'] > 0)].copy()
    
    # 6. Handle data types for categorical columns
    df['state_holiday'] = df['state_holiday'].astype(str)
    
    # 7. Target Scaling (Training only)
    y = None
    if is_training:
        y = np.log1p(df['sales'])
    
    # 8. Feature Selection: Keep only what the model needs
    cols_to_drop = ['sales', 'date', 'open', 'nb_customers_on_day']
    X = df.drop(columns=cols_to_drop, errors='ignore')
    
    return X, y
def run_stability_test(pipeline, X, y):
    print("\n--- 🏁 Starting Cross-Validation (5-Folds) ---")
    # n_jobs=-1 uses all CPU cores for speed
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
    
    print(f"Individual Fold Scores: {scores}")
    print(f"Mean R2 Score: {scores.mean():.4f}")
    print(f"Standard Deviation (Stability): +/- {scores.std():.4f}")
    
    if scores.std() > 0.05:
        print("⚠️ Warning: Your model is somewhat inconsistent across different folds.")
    else:
        print("✅ Stability Passed: The model is consistent across the data.")
def run_model_process(raw_df):
    print("Preprocessing data and isolating features...")
    X, y = preprocess_data(raw_df)
    
    # Identify categorical columns
    cat_features = ['state_holiday', 'store_ID'] 
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ], remainder='passthrough')
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=-1))
    ])
    # 1. Run Stability Check (Cross-Validation)
    run_stability_test(pipeline, X, y)
    
    # 2. Final Training & Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining final XGBoost model...")
    pipeline.fit(X_train, y_train)
    
    score = pipeline.score(X_test, y_test)
    print(f"✅ Final Model R2 Score (Log Scale): {score:.4f}")
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    return pipeline
if __name__ == "__main__":
    sales_df = pd.read_csv('sales.csv', index_col=0)
    run_model_process(sales_df)