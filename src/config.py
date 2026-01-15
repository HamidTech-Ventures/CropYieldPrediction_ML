import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Data File Paths
YIELD_PATH = os.path.join(DATA_DIR, 'yield.csv')
RAIN_PATH = os.path.join(DATA_DIR, 'rainfall.csv')
PESTICIDES_PATH = os.path.join(DATA_DIR, 'pesticides.csv')
TEMP_PATH = os.path.join(DATA_DIR, 'temp.csv')

# Model Save Path
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'crop_yield_rf_model.pkl')
PREPROCESSOR_SAVE_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100  # Number of trees in Random Forest
