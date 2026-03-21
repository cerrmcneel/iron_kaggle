import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

def check_stability(pipeline, X, y):
    print("\n--- 🏁 Running 5-Fold Cross-Validation ---")
    # This splits the data 5 times to see if 0.425 is consistent
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
    print(f"Scores: {scores}")
    print(f"Average R2: {scores.mean():.4f} (Stability: +/- {scores.std():.4f})")
    return scores.mean()



def preprocess_data(df):
    """Consider your previous cleaning steps here"""
    # 1. Subset Creation: Keep only open stores with sales
    df = df[(df['open'] != 0) & (df['sales'] > 0)].copy()

    # 2. Target Scaling: Add sales_log
    df['sales_log'] = np.log1p(df['sales'])

    # 3. Feature Engineering: Date processing
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # 4. Clean up types
    df['state_holiday'] = df['state_holiday'].astype(str)

    return df

def run_model_process(raw_df):
    print("Preprocessing data using notebook logic...")
    df = preprocess_data(raw_df)

    # Define Features and Target (Use sales_log!)
    # We drop columns we don't want the model to 'see'
    X = df.drop(columns=['sales', 'sales_log', 'date', 'Unnamed: 0', 'open', 'nb_customers_on_day'], errors='ignore')
    y = df['sales_log']

    # Identify categorical columns for the Pipeline to handle
    cat_features = ['state_holiday', 'store_ID']
    num_features = [col for col in X.columns if col not in cat_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import cross_val_score

def run_stability_test(pipeline, X, y):
    print("\n--- 🏁 Starting Cross-Validation (5-Folds) ---")
    
    # We use cv=5 to split the data into 5 different 'exams'
    # n_jobs=-1 makes it run faster by using all your computer's "brains"
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
    
    print(f"Individual Fold Scores: {scores}")
    print(f"Mean R2 Score: {scores.mean():.4f}")
    print(f"Standard Deviation: {scores.std():.4f}")
    
    if scores.std() > 0.05:
        print("⚠️ Warning: Your model is inconsistent across different time periods.")
    else:
        print("✅ Stability Passed: The 0.425 score is consistent across the data.")

    print("Training XGBoost on log scale...")
    pipeline.fit(X_train, y_train)

    print(f":white_check_mark: Original Model R2 Score (Log Scale): {pipeline.score(X_test, y_test):.4f}")

    return pipeline

if __name__ == "__main__":
    sales_df = pd.read_csv('sales.csv')
    run_model_process(sales_df)