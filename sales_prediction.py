# ==========================================
# Sales Prediction using Python
# Task 4 - CodeAlpha Internship
# ==========================================

# ------------------------------------------
# Step 1: Import Libraries
# ------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ------------------------------------------
# Step 2: Load Dataset
# ------------------------------------------
# Dataset: Advertising Sales dataset
# Columns: TV, Radio, Newspaper (ad spend), Sales (target)
# Download from: https://www.kaggle.com/datasets/bumba5341/advertisingsalescsv

df = pd.read_csv(r"C:\Users\YUG M PATHAK\PyCharmMiscProject\codealpha projects\Advertising (1).csv")

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET INFO ==========")
print(df.info())

print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# ------------------------------------------
# Step 3: Data Cleaning & Preparation
# ------------------------------------------

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check for missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()

# Drop missing values
df = df.dropna()

print(f"\nShape after cleaning: {df.shape}")

# ------------------------------------------
# Step 4: Exploratory Data Analysis (EDA)
# ------------------------------------------

# Distribution of Sales
plt.figure(figsize=(8, 5))
sns.histplot(df['Sales'], kde=True, color='steelblue')
plt.title("Distribution of Sales")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Advertising spend vs Sales (scatter plots)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
channels = ['TV', 'Radio', 'Newspaper']
colors = ['#e74c3c', '#2ecc71', '#3498db']

for ax, channel, color in zip(axes, channels, colors):
    ax.scatter(df[channel], df['Sales'], alpha=0.6, color=color, edgecolors='white', linewidth=0.5)
    ax.set_xlabel(f"{channel} Ad Spend ($)")
    ax.set_ylabel("Sales")
    ax.set_title(f"{channel} Spend vs Sales")
    ax.grid(True, alpha=0.3)

plt.suptitle("Advertising Channels vs Sales", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Pairplot
sns.pairplot(df, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle("Pairplot of Advertising Data", y=1.02)
plt.tight_layout()
plt.show()

# ------------------------------------------
# Step 5: Feature Engineering
# ------------------------------------------

# Total advertising spend
df['Total_Spend'] = df['TV'] + df['Radio'] + df['Newspaper']

# TV to Radio ratio (dominant channel ratio)
df['TV_Radio_Ratio'] = df['TV'] / (df['Radio'] + 1)

# High TV spend flag
df['High_TV'] = (df['TV'] > df['TV'].median()).astype(int)

print("\n========== FEATURES AFTER ENGINEERING ==========")
print(df.head())

# ------------------------------------------
# Step 6: Define Features and Target
# ------------------------------------------

feature_cols = ['TV', 'Radio', 'Newspaper', 'Total_Spend', 'TV_Radio_Ratio', 'High_TV']
target_col   = 'Sales'

X = df[feature_cols]
y = df[target_col]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

# ------------------------------------------
# Step 7: Train-Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size    : {X_test.shape[0]}")

# ------------------------------------------
# Step 8: Train Multiple Models
# ------------------------------------------

models = {
    "Linear Regression"        : LinearRegression(),
    "Random Forest"            : RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting"        : GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}

print("\n========== MODEL TRAINING & EVALUATION ==========")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # Cross-validation R² score
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

    results[name] = {
        "MAE" : mae,
        "RMSE": rmse,
        "R2"  : r2,
        "CV_R2_mean": cv_scores.mean(),
        "CV_R2_std" : cv_scores.std(),
        "model"     : model,
        "y_pred"    : y_pred,
    }

    print(f"\n--- {name} ---")
    print(f"  MAE        : {mae:.4f}")
    print(f"  RMSE       : {rmse:.4f}")
    print(f"  R²         : {r2:.4f}")
    print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ------------------------------------------
# Step 9: Compare Models Visually
# ------------------------------------------

model_names = list(results.keys())
r2_scores   = [results[m]["R2"] for m in model_names]
mae_scores  = [results[m]["MAE"] for m in model_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(model_names, r2_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
axes[0].set_title("Model Comparison — R² Score")
axes[0].set_xlabel("R² Score")
axes[0].set_xlim(0, 1)
for i, v in enumerate(r2_scores):
    axes[0].text(v + 0.01, i, f"{v:.3f}", va='center')

axes[1].barh(model_names, mae_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
axes[1].set_title("Model Comparison — MAE (lower is better)")
axes[1].set_xlabel("Mean Absolute Error")
for i, v in enumerate(mae_scores):
    axes[1].text(v + 0.01, i, f"{v:.3f}", va='center')

plt.tight_layout()
plt.show()

# ------------------------------------------
# Step 10: Best Model — Actual vs Predicted
# ------------------------------------------

best_model_name = max(results, key=lambda m: results[m]["R2"])
best_result     = results[best_model_name]

print(f"\n========== BEST MODEL: {best_model_name} ==========")

y_pred_best = best_result["y_pred"]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.7, color='steelblue', edgecolors='white')
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title(f"Actual vs Predicted Sales ({best_model_name})")
plt.legend()
plt.tight_layout()
plt.show()

# Residuals plot
residuals = y_test.values - y_pred_best

plt.figure(figsize=(8, 5))
plt.scatter(y_pred_best, residuals, alpha=0.6, color='coral', edgecolors='white')
plt.axhline(0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.tight_layout()
plt.show()

# ------------------------------------------
# Step 11: Feature Importance (Random Forest)
# ------------------------------------------

rf_model = results["Random Forest"]["model"]
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

print("\n========== FEATURE IMPORTANCE ==========")
print(feat_imp)

# ------------------------------------------
# Step 12: Advertising Impact Analysis
# ------------------------------------------

# Simulate: What happens to sales if TV budget increases by 10%?
print("\n========== ADVERTISING IMPACT SIMULATION ==========")

base_input = pd.DataFrame([{
    'TV'            : df['TV'].mean(),
    'Radio'         : df['Radio'].mean(),
    'Newspaper'     : df['Newspaper'].mean(),
    'Total_Spend'   : df['Total_Spend'].mean(),
    'TV_Radio_Ratio': df['TV_Radio_Ratio'].mean(),
    'High_TV'       : 1,
}])

best_model = results[best_model_name]["model"]

for channel in ['TV', 'Radio', 'Newspaper']:
    modified = base_input.copy()
    modified[channel] = modified[channel] * 1.10  # 10% increase
    modified['Total_Spend'] = modified['TV'] + modified['Radio'] + modified['Newspaper']

    base_scaled    = scaler.transform(base_input)
    modified_scaled = scaler.transform(modified)

    base_sales     = best_model.predict(base_scaled)[0]
    modified_sales = best_model.predict(modified_scaled)[0]
    impact         = modified_sales - base_sales

    print(f"  +10% {channel:10s} spend → Sales change: {impact:+.2f} units")

# ------------------------------------------
# Step 13: Key Insights
# ------------------------------------------

print("\n========== KEY INSIGHTS ==========")
print("1. TV advertising has the strongest positive correlation with sales.")
print("2. Radio also contributes significantly to sales outcomes.")
print("3. Newspaper advertising shows the weakest impact on sales.")
print(f"4. Best model: {best_model_name} with R² = {best_result['R2']:.3f}")
print("5. Increasing TV ad spend yields the highest return on investment.")
print("6. Combined advertising spend (Total_Spend) is a strong predictor.")

print("\n========== ANALYSIS COMPLETED ==========")
