import pandas as pd
import sys
from src.preprocessing import DataPreprocessor
from src.model import CropYieldModel

def main():
    print("""
    ========================================================
      CROP YIELD PREDICTION SYSTEM (ML Research Implementation)
    ========================================================
    """)

    # 1. Data Loading & Preprocessing
    try:
        dp = DataPreprocessor()
        df = dp.load_and_clean_data()
        
        X_train, X_test, y_train, y_test = dp.prepare_data(df)
        
        preprocessor_step = dp.create_preprocessor(X_train)
        
    except Exception as e:
        print(f" [ERROR] Data pipeline failed: {e}")
        sys.exit(1)

    # 2. Model Training
    try:
        model = CropYieldModel()
        model.build_pipeline(preprocessor_step)
        model.train(X_train, y_train)
        
    except Exception as e:
        print(f" [ERROR] Model training failed: {e}")
        sys.exit(1)

    # 3. Evaluation
    model.evaluate(X_test, y_test)
    
    # 4. Save Model
    model.save_model()

    # 5. Live Demonstration Prediction
    print(" [DEMO] Running a sample prediction from Test Set...")
    sample_row = X_test.iloc[0:1]
    actual_value = y_test.iloc[0]
    predicted_value = model.predict_sample(sample_row)
    
    print(f"\n Input Data (Area: {sample_row['Area'].values[0]}, Item: {sample_row['Item'].values[0]})")
    print(f" Actual Yield:    {actual_value:.2f} hg/ha")
    print(f" Predicted Yield: {predicted_value:.2f} hg/ha")
    print(f" Difference:      {abs(actual_value - predicted_value):.2f}\n")

if __name__ == "__main__":
    main()
