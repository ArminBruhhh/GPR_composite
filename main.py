import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # هشدارها را موقتاً مخفی کن

from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import KFold  # تغییر از LOOCV به KFold
from sklearn.metrics import r2_score, mean_squared_error

# ===============================
# 1. Raw data
# ===============================
data = [
    ("DW","CU",[40.924,38.362,32.243,25.971,24.262]),
    ("DW","BL",[38.25,35.061,29.963,24.726,22.983]),
    ("DW","SP",[43.815,41.64,36.152,35.091,34.3]),

    ("AS","CU",[40.924,38.631,34.462,32.07,25.004]),
    ("AS","BL",[38.25,34.336,29.62,27.667,23.522]),
    ("AS","SP",[43.815,40.73,38.973,37.084,35.673]),

    ("EW","CU",[40.924,36.014,33.417,20.421,18.461]),
    ("EW","BL",[38.25,33.116,28.689,19.926,19.922]),
    ("EW","SP",[43.815,39.011,35.331,28.117,26.285]),

    ("MW","CU",[40.924,37.941,33.29,28.614,24.622]),
    ("MW","BL",[38.25,34.012,30.633,26.793,24.632]),
    ("MW","SP",[43.815,41.351,40.591,38.554,33.319])
]

time = np.array([1, 15, 30, 60, 90])

# ===============================
# 2. Compute degradation slopes
# ===============================
rows = []
for media, comp, values in data:
    slope = np.polyfit(time, values, 1)[0]
    rows.append([media, comp, slope])

df = pd.DataFrame(rows, columns=["Media", "Composite", "Slope"])
print("Degradation Slopes Table:")
print(df.to_string(index=False))
print("\n" + "="*50)

# ===============================
# 3. Encode categorical variables
# ===============================
try:
    encoder = OneHotEncoder(sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(sparse=False)

X = encoder.fit_transform(df[["Media", "Composite"]])
y = df["Slope"].values

# ===============================
# 4. K-Fold Cross-Validation (به جای LOOCV)
# ===============================
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # فقط 5 بار آموزش

y_true_all = []
y_pred_all = []

# هسته اصلاح‌شده
kernel = RBF(length_scale=1.0) + WhiteKernel(
    noise_level=1e-5, 
    noise_level_bounds=(1e-10, 1e-2)  # محدوده وسیع‌تر
)

for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,  # کاهش تعداد restartها
        random_state=42
    )

    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)
    
    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)
    
    print(f"Fold {fold}: Test indices {test_idx}")

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# ===============================
# 5. Performance metrics
# ===============================
r2 = r2_score(y_true_all, y_pred_all)
rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

print("\n" + "="*50)
print(f"5-Fold CV Results:")
print(f"R²   = {r2:.3f}")
print(f"RMSE = {rmse:.4f} MPa/day")
print("="*50)

# ===============================
# 6. Final model on all data
# ===============================
final_gpr = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,
    n_restarts_optimizer=10,
    random_state=42
)

final_gpr.fit(X, y)
final_pred, final_std = final_gpr.predict(X, return_std=True)

print("\nFinal Model Predictions with Uncertainty:")
results = df.copy()
results["Predicted_Slope"] = final_pred
results["Uncertainty"] = final_std
print(results.to_string(index=False))

# ===============================
# 7. Predicted vs Experimental plot
# ===============================
plt.figure(figsize=(8, 6))
plt.scatter(y_true_all, y_pred_all, s=100, alpha=0.7, edgecolors='k', label='5-Fold CV Points')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')

plt.xlabel("Experimental Slope (MPa/day)", fontsize=12)
plt.ylabel("Predicted Slope (MPa/day)", fontsize=12)
plt.title("Gaussian Process Regression: Predicted vs Experimental Degradation Rates", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# اضافه کردن اعداد R² و RMSE به نمودار
plt.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.4f}', 
         transform=plt.gca().transAxes,
         fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('GPR_Prediction_Plot.png', dpi=300)
plt.show()

# ===============================
# 8. Bar plot for slopes comparison
# ===============================
plt.figure(figsize=(12, 6))
x_pos = np.arange(len(df))
width = 0.35

plt.bar(x_pos - width/2, df["Slope"], width, label='Experimental', alpha=0.8)
plt.bar(x_pos + width/2, results["Predicted_Slope"], width, 
        yerr=results["Uncertainty"], label='GPR Prediction ± σ', alpha=0.8, capsize=5)

plt.xlabel('Media-Composite Combination', fontsize=12)
plt.ylabel('Degradation Rate (MPa/day)', fontsize=12)
plt.title('Comparison of Experimental and Predicted Degradation Rates', fontsize=14)
plt.xticks(x_pos, [f'{m}-{c}' for m, c in zip(df["Media"], df["Composite"])], rotation=45)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('Slope_Comparison_Plot.png', dpi=300)
plt.show()

print("\n✅ Analysis completed successfully!")
print("📊 Two plots have been saved: 'GPR_Prediction_Plot.png' and 'Slope_Comparison_Plot.png'")