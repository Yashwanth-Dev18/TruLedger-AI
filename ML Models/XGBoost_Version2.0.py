import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===============================
# 🔹 LOAD DATASET
# ===============================
df = pd.read_csv("Datasets/Processed/TrainingSet.csv")
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.4f}")

# Features & target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# ===============================
# 🔹 SPLIT DATA (Train/Val/Test)
# ===============================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)
print(f"Train fraud rate: {y_train.mean():.4f} | Val: {y_val.mean():.4f} | Test: {y_test.mean():.4f}")

# ===============================
# 🔹 SCALE FEATURES
# ===============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ===============================
# 🔹 HANDLE IMBALANCE
# ===============================
fraud_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"Fraud ratio (normal:fraud) = {fraud_ratio:.1f}:1")

# ===============================
# 🔹 TRAIN XGBOOST MODEL
# ===============================
xgb_model = XGBClassifier(
    scale_pos_weight=fraud_ratio,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20
)

xgb_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=10
)

# ===============================
# 🔹 THRESHOLD TUNING
# ===============================
y_val_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
thresholds_to_test = [0.1 * i for i in range(1, 10)]
results = []

for threshold in thresholds_to_test:
    y_val_pred = (y_val_proba > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    results.append({'threshold': threshold, 'precision': precision, 'recall': recall, 'f1': f1})

results_df = pd.DataFrame(results)
optimal_idx = results_df['f1'].idxmax()
optimal_threshold = results_df.loc[optimal_idx, 'threshold']
print(f"Optimal threshold = {optimal_threshold:.2f} (max F1-score)")

# ===============================
# 🔹 SAVE MODEL, SCALER, THRESHOLD
# ===============================
joblib.dump(xgb_model, 'xgboost_fraud_model.pkl')
joblib.dump(scaler, 'xgboost_scaler.pkl')
joblib.dump(optimal_threshold, 'optimal_threshold.pkl')
print("✅ Model, scaler, and threshold saved successfully.")

# ===============================
# 🔹 FEATURE IMPORTANCE (optional)
# ===============================
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10,8))
sns.barplot(data=importance_df, y='feature', x='importance')
plt.title('Top 20 Features - XGBoost Importance')
plt.tight_layout()
plt.show()
