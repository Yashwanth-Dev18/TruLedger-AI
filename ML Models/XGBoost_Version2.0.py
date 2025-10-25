import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================
# 1️⃣ Load Dataset
# =========================
df = pd.read_csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/TrainingSet.csv")
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.4f}")

# =========================
# 2️⃣ Split Features and Target
# =========================
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# =========================
# 3️⃣ Train / Validation / Test Split
# =========================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

print(f"Train fraud rate: {y_train.mean():.4f} (n={len(y_train)})")
print(f"Validation fraud rate: {y_val.mean():.4f} (n={len(y_val)})")
print(f"Test fraud rate: {y_test.mean():.4f} (n={len(y_test)})")

# =========================
# 4️⃣ Scale Features
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

fraud_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"Fraud ratio (normal:fraud): {fraud_ratio:.1f}:1")

# =========================
# 5️⃣ Train XGBoost Model
# =========================
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

# =========================
# 6️⃣ Predictions
# =========================
y_train_proba = xgb_model.predict_proba(X_train_scaled)[:, 1]
y_val_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
y_test_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

y_test_pred_default = xgb_model.predict(X_test_scaled)

# =========================
# 7️⃣ Evaluate Default Threshold
# =========================
print("\nDEFAULT THRESHOLD (0.5) - TEST SET RESULTS:")
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_test_pred_default))
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_test_pred_default))

# =========================
# 8️⃣ Feature Importance
# =========================
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, y='feature', x='importance')
plt.title('Top 20 Features - XGBoost Importance')
plt.tight_layout()
plt.show()

print("\n Top 10 Most Important Features:")
print(importance_df.head(10))

# =========================
# 9️⃣ Fine-Tune Threshold
# =========================
thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for threshold in thresholds_to_test:
    y_val_pred = (y_val_proba > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    results.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'false_positives': fp,
        'false_positive_rate': false_positive_rate,
        'frauds_caught': tp,
        'frauds_missed': fn
    })

results_df = pd.DataFrame(results)
print("\n THRESHOLD COMPARISON (Validation Set):")
print(results_df.round(4))

optimal_idx = results_df['f1'].idxmax()
optimal_threshold = results_df.loc[optimal_idx, 'threshold']
print(f"\n OPTIMAL THRESHOLD: {optimal_threshold:.3f}")

# =========================
# 10️⃣ Apply Optimal Threshold to Test Set
# =========================
y_test_optimal = (y_test_proba > optimal_threshold).astype(int)
print("OPTIMAL CONFUSION MATRIX (Test Set):")
print(confusion_matrix(y_test, y_test_optimal))
print("\nOPTIMAL CLASSIFICATION REPORT (Test Set):")
print(classification_report(y_test, y_test_optimal))

# =========================
# 11️⃣ Save Model, Scaler Params, Threshold
# =========================
# Save XGBoost model (JSON format)
xgb_model.save_model('xgboost_fraud_model.json')

# Save Scaler parameters
scaler_params = {
    'mean': scaler.mean_,
    'scale': scaler.scale_
}
joblib.dump(scaler_params, 'xgboost_scaler_params.pkl')

# Save optimal threshold
joblib.dump(optimal_threshold, 'optimal_threshold.pkl')

print("\n💾 Models saved:")
print(" - xgboost_fraud_model.json")
print(" - xgboost_scaler_params.pkl")
print(f" - optimal_threshold.pkl (threshold: {optimal_threshold:.3f})")
