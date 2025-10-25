import pandas as pd
import numpy as np
import joblib
import json

def run_fraud_detection(processed_file_path):
    """Run fraud detection and save outputs for Streamlit visualization"""

    # Load model, scaler, threshold
    try:
        model = joblib.load('xgboost_fraud_model.pkl')
        scaler = joblib.load('xgboost_scaler.pkl')
        threshold = joblib.load('optimal_threshold.pkl')
        print(f"✅ Model loaded (threshold={threshold:.2f})")
    except FileNotFoundError:
        print("❌ Model files not found! Run training first.")
        return False

    # Load processed data
    df = pd.read_csv(processed_file_path)
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # Scale & predict
    X_scaled = scaler.transform(X)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba > threshold).astype(int)

    # Save detected frauds
    fraud_indices = np.where(y_pred == 1)[0]
    fraud_transactions = X.iloc[fraud_indices].copy()
    fraud_transactions['fraud_probability'] = y_proba[fraud_indices]
    fraud_transactions['is_fraud_predicted'] = 1
    fraud_transactions['is_fraud_actual'] = y.iloc[fraud_indices].values
    fraud_transactions.to_csv('detected_fraud_transactions.csv', index=False)

    # Prepare visualization data
    actual_fraud = X[y == 1]
    job_columns = [col for col in actual_fraud.columns if col.startswith('JOBctg_')]
    txn_columns = [col for col in actual_fraud.columns if col.startswith('TXNctg_')]
    dob_columns = [col for col in actual_fraud.columns if col.startswith('dob_')]

    viz_data = {
        'job_categories': actual_fraud[job_columns].sum().sort_values(ascending=False).head(5).to_dict(),
        'transaction_categories': actual_fraud[txn_columns].sum().sort_values(ascending=False).head(5).to_dict(),
        'age_groups': actual_fraud[dob_columns].sum().sort_values(ascending=False).to_dict(),
        'amount_analysis': {
            'fraud_avg': float(actual_fraud['amt'].mean()),
            'normal_avg': float(X[y == 0]['amt'].mean()),
            'increase_pct': float(((actual_fraud['amt'].mean() - X[y == 0]['amt'].mean()) / X[y == 0]['amt'].mean()) * 100)
        },
        'time_analysis': {
            'fraud_avg_hour': float(actual_fraud['txn_time'].mean()),
            'normal_avg_hour': float(X[y == 0]['txn_time'].mean())
        },
        'model_performance': {
            'fraud_detected': int(y_pred.sum()),
            'total_transactions': len(y_pred),
            'detection_rate': float(y_pred.sum()/len(y_pred))
        }
    }

    with open('app_visualization_data.json', 'w') as f:
        json.dump(viz_data, f, indent=2)

    print(f"✅ Fraud detection complete. {len(fraud_indices)} transactions flagged.")
    return True

# For standalone testing
if __name__ == "__main__":
    test_file = "Uploaded_Datasets/Processed/ProcessedTransactionLogs-1.csv"
    run_fraud_detection(test_file)
