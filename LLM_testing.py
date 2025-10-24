import pandas as pd
import numpy as np
import joblib
from langchain_groq import ChatGroq

class TruLedgerExplainer:
    def __init__(self, groq_api_key):
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
        # Load your trained model
        self.xgb_model = joblib.load('xgboost_fraud_model.pkl')
        self.scaler = joblib.load('xgboost_scaler.pkl') 
        self.optimal_threshold = joblib.load('optimal_threshold.pkl')
        
        # Store feature names from training
        self.feature_names = None
        self._load_feature_names()
    
    def _load_feature_names(self):
        """Get the feature names that the model was trained on"""
        try:
            # Load original training data to get feature names
            df_train = pd.read_csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/TrainingSet.csv")
            self.feature_names = df_train.drop('is_fraud', axis=1).columns.tolist()
            print(f"✅ Model expects {len(self.feature_names)} features: {self.feature_names[:5]}...")
        except Exception as e:
            print(f"⚠️ Could not load feature names: {e}")
            # Fallback: assume all columns except 'is_fraud'
            df_train = pd.read_csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/TrainingSet.csv")
            self.feature_names = [col for col in df_train.columns if col != 'is_fraud']
    
    def predict_fraud(self, transaction_row):
        """Make fraud prediction - FIXED VERSION"""
        # Extract only the features the model was trained on (exclude 'is_fraud')
        if self.feature_names:
            # Convert to DataFrame to use column names
            transaction_df = pd.DataFrame([transaction_row])
            features_data = transaction_df[self.feature_names].values
        else:
            # Fallback: assume 'is_fraud' is first column
            features_data = transaction_row[1:].reshape(1, -1)  # Skip first column (is_fraud)
        
        # Scale the features
        scaled_data = self.scaler.transform(features_data)
        
        # Get probability
        fraud_probability = self.xgb_model.predict_proba(scaled_data)[0, 1]
        
        # Apply optimal threshold
        is_fraud = fraud_probability > self.optimal_threshold
        
        return {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_probability),
            'threshold': float(self.optimal_threshold)
        }
    
    def extract_transaction_context(self, transaction_row):
        """Convert processed features back to human-readable context"""
        # Convert to Series for easier handling
        if not isinstance(transaction_row, pd.Series):
            transaction_series = pd.Series(transaction_row, index=self._get_all_column_names())
        else:
            transaction_series = transaction_row
        
        context = {}
        
        # Basic transaction info
        context['amount'] = transaction_series['amt']
        context['time_hour'] = int(transaction_series['txn_time'])
        
        # Time period
        if context['time_hour'] < 5:
            context['time_period'] = "late night"
        elif context['time_hour'] < 12:
            context['time_period'] = "morning"
        elif context['time_hour'] < 17:
            context['time_period'] = "afternoon"
        elif context['time_hour'] < 22:
            context['time_period'] = "evening"
        else:
            context['time_period'] = "night"
        
        # Extract category
        category_cols = [col for col in transaction_series.index if col.startswith('TXNctg_')]
        for col in category_cols:
            if transaction_series[col] == 1:
                context['category'] = col.replace('TXNctg_', '').replace('_', ' ').title()
                break
        else:
            context['category'] = "Unknown"
        
        # Extract state
        state_cols = [col for col in transaction_series.index if col.startswith('state_')]
        for col in state_cols:
            if transaction_series[col] == 1:
                context['state'] = col.replace('state_', '')
                break
        else:
            context['state'] = "Unknown"
        
        # User behavior
        context['user_avg_amount'] = transaction_series['avg_txn_amt']
        context['user_avg_time'] = int(transaction_series['avg_txn_time'])
        context['user_amount_variability'] = transaction_series['stddev_txn_amt']
        context['user_avg_distance'] = transaction_series['avg_merchant_distance']
        
        # Calculate anomalies
        context['amount_ratio'] = context['amount'] / context['user_avg_amount'] if context['user_avg_amount'] > 0 else 0
        context['time_deviation'] = abs(context['time_hour'] - context['user_avg_time'])
        
        return context
    
    def _get_all_column_names(self):
        """Get all column names from the processed dataset"""
        df = pd.read_csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/TrainingSet.csv")
        return df.columns.tolist()
    
    def generate_explanation(self, transaction_context, prediction_result):
        """Generate human-readable fraud explanation"""
        
        prompt = f"""
You are TruLedger AI, a professional fraud detection analyst. Explain why this transaction was flagged in clear, simple language.

TRANSACTION:
- Amount: ${transaction_context['amount']:.2f}
- Time: {transaction_context['time_hour']}:00 ({transaction_context['time_period']})
- Category: {transaction_context['category']}
- Location: {transaction_context['state']}

USER'S NORMAL PATTERN:
- Average Amount: ${transaction_context['user_avg_amount']:.2f}
- Typical Time: {transaction_context['user_avg_time']}:00
- Amount Variability: ${transaction_context['user_amount_variability']:.2f}

ANOMALIES:
- Amount is {transaction_context['amount_ratio']:.1f}x user's average
- Time difference: {transaction_context['time_deviation']} hours from normal
- Fraud Probability: {prediction_result['fraud_probability']:.1%}

Decision: {"FRAUDULENT" if prediction_result['is_fraud'] else "LEGITIMATE"}

Provide a short explanation (2-3 sentences) focusing on the main risk factors:
"""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Explanation unavailable. Error: {str(e)}"

# Simple usage
def main():
    # Initialize
    explainer = TruLedgerExplainer("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")
    
    # Load your data
    df = pd.read_csv("c:/Users/hp/LNU/TruLedger-AI/Datasets/Processed/TrainingSet.csv")
    
    # Test with first transaction
    sample_transaction = df.iloc[0]
    
    # Get prediction - pass the entire row, the method will handle feature selection
    prediction = explainer.predict_fraud(sample_transaction)
    
    # Get context
    context = explainer.extract_transaction_context(sample_transaction)
    
    # Get explanation
    explanation = explainer.generate_explanation(context, prediction)
    
    print("🔍 TRULEDGER FRAUD ANALYSIS")
    print("=" * 50)
    print(f"Amount: ${context['amount']:.2f}")
    print(f"Time: {context['time_hour']}:00 ({context['time_period']})")
    print(f"Category: {context['category']}")
    print(f"Location: {context['state']}")
    print(f"Fraud Probability: {prediction['fraud_probability']:.1%}")
    print(f"Decision: {'🚨 FRAUD' if prediction['is_fraud'] else '✅ LEGITIMATE'}")
    print(f"\n🤖 EXPLANATION:")
    print(explanation)

# yeah
if __name__ == "__main__":
    main()