# 🌾 Crop Yield Prediction Using Machine Learning
**Research Implementation | Random Forest Regressor**

## 1. Project Overview
This project uses Machine Learning to predict how much crop (in hectograms per hectare) a farmer will harvest based on:
1.  **Rainfall** (mm/year)
2.  **Temperature** (avg degrees)
3.  **Pesticide Use** (tonnes)
4.  **Country & Crop Type**

**Model Used:** Random Forest Regressor (Ensemble Learning).
**Accuracy Achieved:** ~98.7% ($R^2$ Score).

---

## 2. Setup Instructions (Do this BEFORE class)

### Step 1: Prepare Data
Ensure the `data/` folder contains these 4 files:
* `pesticides.csv`
* `rainfall.csv`
* `temp.csv`
* `yield.csv`

*(Note: If you only have `pesticides.csv`, download the dataset from the Kaggle link provided in the project source or use the `crop-yield-eda-viz.ipynb` data sources).*

### Step 2: Install Python Libraries
Open your terminal/command prompt in this folder and run:
```bash
pip install -r requirements.txt
