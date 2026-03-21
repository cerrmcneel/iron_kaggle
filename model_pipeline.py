import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor 

def run_model_process(sales_df):
    print("Preparing data...")
    
    # 1. Feature Engineering
    df = sales_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['state_holiday'] = df['state_holiday'].astype(str)
    
    # 2. Define Features and Target
    # We drop 'sales' (target), 'date' (already split), and index columns
    X = df.drop(columns=['sales', 'date', 'Unnamed: 0'], errors='ignore')
    y = df['sales']
    
    # 3. Identify categorical and numerical columns
    # Your teammate is focusing on these for One-Hot Encoding
    cat_features = ['state_holiday', 'store_ID'] 
    num_features = [col for col in X.columns if col not in cat_features]
    
    # 4. The Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # 5. The Pipeline: Preprocessing + XGBoost
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1))
    ])
    
    # 6. Train/Test Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost...")
    pipeline.fit(X_train, y_train)
    
    score = pipeline.score(X_test, y_test)
    print(f"✅ Model R2 Score: {score:.4f}")
    
    # 7. Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("✅ model.pkl saved.")
    
    return pipeline

# If you run this script directly, it executes:
if __name__ == "__main__":
    sales_df = pd.read_csv('sales.csv')
    run_model_process(sales_df)