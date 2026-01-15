import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from . import config

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None

    def load_and_clean_data(self):
        print(" [INFO] Loading datasets...")
        
        # 1. Load Yield Data
        try:
            df_yield = pd.read_csv(config.YIELD_PATH)
            # Rename Value to yield and drop unnecessary columns (from notebook analysis)
            df_yield = df_yield.rename(index=str, columns={"Value": "hg/ha_yield"})
            cols_to_drop = ['Year Code', 'Element Code', 'Element', 'Area Code', 'Domain Code', 'Domain', 'Unit', 'Item Code']
            df_yield = df_yield.drop([c for c in cols_to_drop if c in df_yield.columns], axis=1)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing file: {config.YIELD_PATH}")

        # 2. Load Rainfall Data
        try:
            df_rain = pd.read_csv(config.RAIN_PATH)
            df_rain = df_rain.rename(index=str, columns={" Area": 'Area'})
            # Convert rainfall to numeric, coercing errors
            df_rain['average_rain_fall_mm_per_year'] = pd.to_numeric(df_rain['average_rain_fall_mm_per_year'], errors='coerce')
            df_rain = df_rain.dropna()
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing file: {config.RAIN_PATH}")

        # 3. Load Pesticides Data
        try:
            df_pes = pd.read_csv(config.PESTICIDES_PATH)
            df_pes = df_pes.rename(index=str, columns={"Value": "pesticides_tonnes"})
            df_pes = df_pes.drop(['Element', 'Domain', 'Unit', 'Item'], axis=1, errors='ignore')
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing file: {config.PESTICIDES_PATH}")

        # 4. Load Temperature Data
        try:
            df_temp = pd.read_csv(config.TEMP_PATH)
            df_temp = df_temp.rename(index=str, columns={"year": "Year", "country": "Area"})
            df_temp = df_temp.dropna()
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing file: {config.TEMP_PATH}")

        # --- Merging Dataframes (Logic from Notebook) ---
        print(" [INFO] Merging datasets...")
        
        # Merge Yield + Rain
        yield_df = pd.merge(df_yield, df_rain, on=['Year', 'Area'])
        
        # Merge + Pesticides
        yield_df = pd.merge(yield_df, df_pes, on=['Year', 'Area'])
        
        # Merge + Temp
        yield_df = pd.merge(yield_df, df_temp, on=['Area', 'Year'])
        
        print(f" [INFO] Final Dataset Shape: {yield_df.shape}")
        return yield_df

    def create_preprocessor(self, X):
        """Creates a Scikit-Learn ColumnTransformer for OneHotEncoding and Scaling"""
        
        # Identify Categorical and Numerical Columns
        categorical_features = ['Area', 'Item']
        numerical_features = ['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp'] 
        # Note: 'Year' is often treated as numeric or dropped depending on strategy. 
        # The paper suggests temporal dependencies, but standard RF handles year as a feature okay.
        
        # Pipeline for Categorical
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Pipeline for Numerical
        numerical_transformer = StandardScaler()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough' # Keep Year if passed
        )
        return self.preprocessor

    def prepare_data(self, df):
        """Splits features and target, prepares pipeline"""
        
        # Target variable
        y = df['hg/ha_yield']
        
        # Feature variables (Drop target and Year if strictly not wanted, but Year is useful trend info)
        X = df.drop(['hg/ha_yield'], axis=1) 
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
        )
        
        return X_train, X_test, y_train, y_test
