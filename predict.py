import pandas as pd
import numpy as np
import pickle
from model_pipeline import preprocess_data

# The file they provide at 16:00
VAL_FILE = 'validation.csv' 
# The historical data for lag features
TRAIN_FILE = 'sales.csv'

def load_csv_safely(filepath):
    """Try to load a CSV with common encodings."""
    for enc in ['utf-8', 'utf-16', 'latin1']:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    return pd.read_csv(filepath) # Final attempt with default

def generate_predictions():
    print(f"Loading {VAL_FILE} and historical data from {TRAIN_FILE}...")
    df_val = load_csv_safely(VAL_FILE)
    if 'sales' in df_val.columns:
        df_val = df_val.drop(columns=['sales'])
    
    try:
        df_train = pd.read_csv(TRAIN_FILE)
    except FileNotFoundError:
        print(f"Error: {TRAIN_FILE} not found. We need historical sales for lag features!")
        return

    # 1. Combine datasets to compute time-based features correctly
    print("Combining datasets to compute historical lags...")
    # Add a unique temp_id to EVERYTHING to track rows after sorting/dropping
    df_val['temp_id'] = range(len(df_val))
    df_train['temp_id'] = -1 # indicator for training rows
    
    df_combined = pd.concat([df_train, df_val], ignore_index=True)
    
    # 2. Preprocess the combined dataset
    print("Preprocessing data and engineering features...")
    # We pass a copy to avoid side effects
    X_combined, _ = preprocess_data(df_combined.copy(), is_training=False)
    
    # 3. Isolate validation rows
    # X_combined still has 'temp_id' because it's not in the drop list
    print("Isolating validation predictors...")
    X_val_final = X_combined[X_combined['temp_id'] >= 0].sort_values('temp_id').copy()
    
    # Drop our tracking ID
    X_val_final = X_val_final.drop(columns=['temp_id'], errors='ignore')
    
    # 4. Predict
    print("Loading model.pkl...")
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Error: model.pkl not found! Please run model_pipeline.py first.")
        return
    
    print("Generating predictions (Log Space)...")
    log_preds = model.predict(X_val_final)
    
    print("Converting Log predictions back to Dollars...")
    real_preds = np.expm1(log_preds)
    
    # 5. Format for submission
    # Identify the ID column (could be 'id', 'Unnamed: 0', or just index)
    id_col = 'Unnamed: 0' if 'Unnamed: 0' in df_val.columns else ('id' if 'id' in df_val.columns else None)
    
    output = pd.DataFrame({
        'index': df_val[id_col] if id_col else df_val.index,
        'sales': real_preds
    })
    
    # Set sales to 0 for closed days
    print("Ensuring closed stores predict 0 sales...")
    closed_mask = df_val['open'] == 0
    output.loc[closed_mask, 'sales'] = 0
    
    output.to_csv('predictions.csv', index=False)
    print("✅ SUCCESS: 'predictions.csv' is ready for submission!")

if __name__ == "__main__":
    generate_predictions()
