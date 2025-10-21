import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Loading Dataset
df = pd.read_csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/TrainingSet.csv")
print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['is_fraud'].mean():.4f}")

# Splitting Features and Target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Training, Cross Validation, and Testing Sets Split (3-way split)
print("\n SPLITTING DATA:")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,  # 0.25 * 0.8 = 0.2 --> 20% validation
    stratify=y_temp,
    random_state=42
)

print(f"Train fraud rate: {y_train.mean():.4f} (n={len(y_train)})")
print(f"Validation fraud rate: {y_val.mean():.4f} (n={len(y_val)})") 
print(f"Test fraud rate: {y_test.mean():.4f} (n={len(y_test)})")

# Scaling Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


fraud_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"Fraud ratio (normal:fraud): {fraud_ratio:.1f}:1")

# Training XGBoost Model with Fraud/Anomaly Focus
print("\n Training XGBoost...")
xgb_model = XGBClassifier(
    scale_pos_weight=fraud_ratio,  # Critical for imbalance! 
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
    eval_set=[(X_val_scaled, y_val)],  # Using validation for early stopping
    verbose=10
)

# Getting Predictions for ALL sets
y_train_proba = xgb_model.predict_proba(X_train_scaled)[:, 1]
y_val_proba = xgb_model.predict_proba(X_val_scaled)[:, 1]
y_test_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Default threshold (0.5) predictions
y_test_pred_default = xgb_model.predict(X_test_scaled)

# Evaluating Performance (Default Threshold 0.5 on TEST set)
print("\nDEFAULT THRESHOLD (0.5) - TEST SET RESULTS:")
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_test_pred_default))
print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_test_pred_default))

# Feature Importance
feature_importance = xgb_model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, y='feature', x='importance')
plt.title('Top 20 Features - XGBoost Importance')
plt.tight_layout()
plt.show()

print("\n Top 10 Most Important Features:")
print(importance_df.head(10))

# FINE-TUNING THRESHOLDS USING VALIDATION SET
print("\n FINE-TUNING THRESHOLDS (Using Validation Set):")

thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for threshold in thresholds_to_test:
    y_val_pred = (y_val_proba > threshold).astype(int)
    
    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = cm.ravel()
    
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

# Display results
results_df = pd.DataFrame(results)
print("\n THRESHOLD COMPARISON (Validation Set):")
print(results_df.round(4))

# Find optimal threshold (balancing precision and recall)
optimal_idx = results_df['f1'].idxmax()
optimal_threshold = results_df.loc[optimal_idx, 'threshold']

print(f"\n OPTIMAL THRESHOLD: {optimal_threshold:.3f} (maximizes F1-score)")
print(f"   At this threshold:")
print(f"   - Precision: {results_df.loc[optimal_idx, 'precision']:.3f}")
print(f"   - Recall: {results_df.loc[optimal_idx, 'recall']:.3f}")
print(f"   - F1-score: {results_df.loc[optimal_idx, 'f1']:.3f}")
print(f"   - False Positives: {results_df.loc[optimal_idx, 'false_positives']}")
print(f"   - Frauds Caught: {results_df.loc[optimal_idx, 'frauds_caught']}/{results_df.loc[optimal_idx, 'frauds_caught'] + results_df.loc[optimal_idx, 'frauds_missed']}")

# Visualize trade-off
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(results_df['threshold'], results_df['precision'], marker='o', label='Precision')
plt.plot(results_df['threshold'], results_df['recall'], marker='o', label='Recall')
plt.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7, label='Optimal')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs Threshold (Validation Set)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(results_df['threshold'], results_df['false_positive_rate'], marker='o', color='orange')
plt.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.7, label='Optimal')
plt.xlabel('Threshold')
plt.ylabel('False Positive Rate')
plt.title('False Positive Rate vs Threshold (Validation Set)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# APPLYING OPTIMAL THRESHOLD TO TEST SET
print(f"\n APPLYING OPTIMAL THRESHOLD ({optimal_threshold:.3f}) TO TEST SET:")
y_test_optimal = (y_test_proba > optimal_threshold).astype(int)

print(" OPTIMAL CONFUSION MATRIX (Test Set):")
print(confusion_matrix(y_test, y_test_optimal))
print("\n OPTIMAL CLASSIFICATION REPORT (Test Set):")
print(classification_report(y_test, y_test_optimal))

# Comparison with default threshold
print("\n COMPARISON: Default vs Optimal Threshold")
default_cm = confusion_matrix(y_test, y_test_pred_default)
optimal_cm = confusion_matrix(y_test, y_test_optimal)

print(f"Default (0.5)  - Frauds Caught: {default_cm[1,1]}/{default_cm[1,0] + default_cm[1,1]}, False Positives: {default_cm[0,1]}")
print(f"Optimal ({optimal_threshold:.3f}) - Frauds Caught: {optimal_cm[1,1]}/{optimal_cm[1,0] + optimal_cm[1,1]}, False Positives: {optimal_cm[0,1]}")

# Saving the Model and Optimal Threshold
joblib.dump(xgb_model, 'xgboost_fraud_model.pkl')
joblib.dump(scaler, 'xgboost_scaler.pkl')
joblib.dump(optimal_threshold, 'optimal_threshold.pkl')

print(f"\n💾 Models saved:")
print("   - 'xgboost_fraud_model.pkl'")
print("   - 'xgboost_scaler.pkl'") 
print(f"   - 'optimal_threshold.pkl' (threshold: {optimal_threshold:.3f})")

# FINAL MODEL READY FOR LLM INTEGRATION AND DEPLOYMENT