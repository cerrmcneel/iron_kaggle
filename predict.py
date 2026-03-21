import pandas as pd
import numpy as np
import pickle

VAL_FILE = 'validation_for students.csv' 

def generate_predictions():
    print(f"🚀 Loading {VAL_FILE}...")
    try:
        df_val = pd.read_csv(VAL_FILE)
    except FileNotFoundError:
        print(f"❌ Error: {VAL_FILE} not found!")
        return

    # 1. Preprocessing (Must match what the model expects!)
    print("🛠️ Preprocessing validation data...")
    df_val['date'] = pd.to_datetime(df_val['date'], dayfirst=True)
    
    # ADD THESE THREE LINES:
    df_val['year'] = df_val['date'].dt.year    # This fixes the current error
    df_val['month'] = df_val['date'].dt.month
    df_val['day'] = df_val['date'].dt.day
    
    # Safety fix: if your model was trained with 'promo' instead of 'promotion'
    if 'promotion' in df_val.columns and 'promo' not in df_val.columns:
        df_val['promo'] = df_val['promotion']

    df_val['state_holiday'] = df_val['state_holiday'].astype(str)

    # 2. Load the trained 'Engine'
    print("📂 Loading model.pkl...")
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 3. Predict 
    print("🔮 Generating predictions (Log Space)...")
    log_preds = model.predict(df_val)
    
    # 4. Convert Log back to Dollars
    print("💰 Converting Log predictions back to Dollars...")
    real_preds = np.expm1(log_preds)
    
    # 5. Rule: If store is closed, sales MUST be 0
    real_preds = np.where(df_val['open'] == 0, 0, real_preds)
    
    # 6. Format for submission
    # Using 'index' because that's what the teacher's file has
    output = pd.DataFrame({
        'index': df_val['index'], 
        'sales': real_preds
    })
    
    output['sales'] = output['sales'].clip(lower=0)
    output.to_csv('predictions.csv', index=False)
    print("✅ SUCCESS: 'predictions.csv' is ready!")

if __name__ == "__main__":
    generate_predictions()