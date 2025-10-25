import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler

def run_fraud_detection(processed_file_path):
    
    # Loading the trained model and components
    try:
        model = joblib.load('xgboost_fraud_model.pkl')
        scaler = joblib.load('xgboost_scaler.pkl') 
        optimal_threshold = joblib.load('optimal_threshold.pkl')
        print(f"✅ Model loaded successfully (threshold: {optimal_threshold:.3f})")
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("📝 Please ensure these files are in your deployment:")
        print("   - xgboost_fraud_model.pkl")
        print("   - xgboost_scaler.pkl")
        print("   - optimal_threshold.pkl")
        return False
    except Exception as e:
        print(f"❌ Error loading model files: {e}")
        return False

    try:
        # Loading processed test data
        print(f"Loading processed data: {processed_file_path}")
        df = pd.read_csv(processed_file_path)
        print(f"Dataset shape: {df.shape}")

        # Prepare features
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']

        print(f"Fraud rate in dataset: {y.mean():.4f}")

        # Scale features
        print("\n Scaling features...")
        X_scaled = scaler.transform(X)

        # Get predictions
        print(" Running fraud detection predictions...")
        y_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_proba > optimal_threshold).astype(int)

        print(f" Predictions complete:")
        print(f"   - Fraud probability range: {y_proba.min():.3f} - {y_proba.max():.3f}")
        print(f"   - Transactions flagged as fraud: {y_pred.sum()} ({y_pred.sum()/len(y_pred)*100:.2f}%)")

        # ==============================================
        # 🎯 IDENTIFYING FRAUD TRANSACTIONS
        # ==============================================

        fraud_indices = np.where(y_pred == 1)[0]
        print(f" Found {len(fraud_indices)} fraudulent transactions")

        # Create fraud transactions dataset
        fraud_transactions = X.iloc[fraud_indices].copy()
        fraud_transactions['fraud_probability'] = y_proba[fraud_indices]
        fraud_transactions['is_fraud_predicted'] = 1
        fraud_transactions['is_fraud_actual'] = y.iloc[fraud_indices].values
        fraud_transactions['transaction_amount'] = fraud_transactions['amt']
        fraud_transactions['transaction_hour'] = fraud_transactions['txn_time']

        # Sort by fraud probability
        fraud_transactions = fraud_transactions.sort_values('fraud_probability', ascending=False)

        # Save fraud transactions
        fraud_transactions.to_csv('detected_fraud_transactions.csv', index=False)
        print("Saved fraud transactions to 'detected_fraud_transactions.csv'")

        # =======================================
        # PREPARING DATA FOR APP VISUALIZATIONS
        # =======================================

        actual_fraud_mask = (y == 1)
        actual_fraud_transactions = X[actual_fraud_mask]

        print(f"Analyzing patterns across {len(actual_fraud_transactions)} actual fraud cases...")

        # Job categories involved in fraud
        job_columns = [col for col in actual_fraud_transactions.columns if col.startswith('JOBctg_')]
        job_fraud_counts = actual_fraud_transactions[job_columns].sum().sort_values(ascending=False).head(5)

        # Transaction categories involved in fraud  
        txn_columns = [col for col in actual_fraud_transactions.columns if col.startswith('TXNctg_')]
        txn_fraud_counts = actual_fraud_transactions[txn_columns].sum().sort_values(ascending=False).head(5)

        # Age groups involved in fraud
        dob_columns = [col for col in actual_fraud_transactions.columns if col.startswith('dob_')]
        dob_fraud_counts = actual_fraud_transactions[dob_columns].sum().sort_values(ascending=False)

        # Amount analysis
        fraud_amount_avg = actual_fraud_transactions['amt'].mean()
        normal_amount_avg = X[y == 0]['amt'].mean()
        amount_increase_pct = ((fraud_amount_avg - normal_amount_avg) / normal_amount_avg) * 100

        # Time analysis
        fraud_time_avg = actual_fraud_transactions['txn_time'].mean()
        normal_time_avg = X[y == 0]['txn_time'].mean()

        # Save visualization data
        viz_data = {
            'job_categories': job_fraud_counts.to_dict(),
            'transaction_categories': txn_fraud_counts.to_dict(),
            'age_groups': dob_fraud_counts.to_dict(),
            'amount_analysis': {
                'fraud_avg': float(fraud_amount_avg),
                'normal_avg': float(normal_amount_avg),
                'increase_pct': float(amount_increase_pct)
            },
            'time_analysis': {
                'fraud_avg_hour': float(fraud_time_avg),
                'normal_avg_hour': float(normal_time_avg)
            },
            'model_performance': {
                'fraud_detected': int(y_pred.sum()),
                'total_transactions': len(y_pred),
                'detection_rate': float(y_pred.sum()/len(y_pred))
            }
        }

        with open('app_visualization_data.json', 'w') as f:
            json.dump(viz_data, f, indent=2)

        print(" Saved visualization data to 'app_visualization_data.json'")
        return True
        
    except Exception as e:
        print(f"❌ Error in fraud detection: {e}")
        return False

# For standalone testing
if __name__ == "__main__":
    # Test with a processed file
    test_processed_file = "c:/Users/hp/LNU/TruLedger-AI/Uploaded_Datasets/Processed/ProcessedTransactionLogs-1.csv"
    run_fraud_detection(test_processed_file)