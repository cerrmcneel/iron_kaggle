import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor 

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
    
    print("Training XGBoost on log scale...")
    pipeline.fit(X_train, y_train)
    
    print(f"✅ Original Model R2 Score (Log Scale): {pipeline.score(X_test, y_test):.4f}")
    
    return pipeline

if __name__ == "__main__":
    sales_df = pd.read_csv('sales.csv')
    run_model_process(sales_df)
