import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Try to import the new metric for newer sklearn versions, otherwise fallback
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None

from . import config

class CropYieldModel:
    def __init__(self):
        self.model = None

    def build_pipeline(self, preprocessor):
        """Builds a pipeline with Preprocessor -> Random Forest"""
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=config.N_ESTIMATORS, 
                                                random_state=config.RANDOM_STATE,
                                                n_jobs=-1)) # Use all CPU cores
        ])

    def train(self, X_train, y_train):
        print(f" [INFO] Training Random Forest Model with {config.N_ESTIMATORS} trees...")
        self.model.fit(X_train, y_train)
        print(" [INFO] Training Complete.")

    def evaluate(self, X_test, y_test):
        print(" [INFO] Evaluating Model...")
        predictions = self.model.predict(X_test)
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        # FIX: Handle RMSE calculation for different sklearn versions
        if root_mean_squared_error is not None:
            # For newer sklearn versions (1.4+)
            rmse = root_mean_squared_error(y_test, predictions)
        else:
            # For older sklearn versions, manually calculate sqrt of MSE
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
        
        print("\n" + "="*40)
        print(" MODEL PERFORMANCE METRICS")
        print("="*40)
        print(f" R² Score (Accuracy) : {r2:.4f} ({r2*100:.2f}%)")
        print(f" MAE (Mean Abs Error): {mae:.2f}")
        print(f" RMSE (Root Mean Sq): {rmse:.2f}")
        print("="*40 + "\n")
        
        return r2, mae, rmse

    def save_model(self):
        import os
        if not os.path.exists(config.MODEL_DIR):
            os.makedirs(config.MODEL_DIR)
            
        joblib.dump(self.model, config.MODEL_SAVE_PATH)
        print(f" [INFO] Model saved to {config.MODEL_SAVE_PATH}")

    def predict_sample(self, sample_data):
        """Makes a prediction for a single DataFrame row"""
        prediction = self.model.predict(sample_data)
        return prediction[0]
