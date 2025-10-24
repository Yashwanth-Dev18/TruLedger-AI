import pandas as pd
import numpy as np
import json
import os
from langchain_groq import ChatGroq
import re

class FraudExplainer:
    def __init__(self, groq_api_key=None):
        """Initialize the LLM for fraud explanations"""
        if groq_api_key is None:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",  # Fast and cost-effective
            temperature=0.1,  # Low temperature for consistent explanations
            max_tokens=500
        )
        
        # Load the feature mapping for human-readable names
        self.feature_map = self._create_feature_mapping()
        
        # System prompt for consistent fraud explanations
        self.system_prompt = """You are a financial fraud analyst AI. Your task is to explain why a credit card transaction was flagged as fraudulent based on the transaction details and anomaly patterns.

IMPORTANT GUIDELINES:
1. Be concise and professional (2-3 sentences max)
2. Focus on the most significant anomalies
3. Use specific numbers and comparisons
4. Explain in plain English, not technical terms
5. Always mention the transaction amount and key anomalies
6. Structure: Start with the amount, then explain top 2-3 reasons

CRITICAL: Only use the provided transaction features. Do not invent or assume any additional information."""
    
    def _create_feature_mapping(self):
        """Create human-readable names for encoded features"""
        return {
            # Transaction categories
            'TXNctg_gas_transport': 'Gas/Transport',
            'TXNctg_grocery_pos': 'Grocery Store Purchase',
            'TXNctg_shopping_net': 'Online Shopping',
            'TXNctg_food_dining': 'Food/Dining',
            'TXNctg_entertainment': 'Entertainment',
            'TXNctg_travel': 'Travel',
            'TXNctg_misc_net': 'Online Miscellaneous',
            'TXNctg_kids_pets': 'Kids/Pets',
            'TXNctg_grocery_net': 'Online Grocery',
            'TXNctg_personal_care': 'Personal Care',
            'TXNctg_health_fitness': 'Health/Fitness',
            'TXNctg_home': 'Home Goods',
            'TXNctg_misc_pos': 'In-Store Miscellaneous',
            'TXNctg_shopping_pos': 'In-Store Shopping',
            
            # Job categories
            'JOBctg_Business_&_Management': 'Business/Management',
            'JOBctg_Healthcare_&_Medical': 'Healthcare/Medical',
            'JOBctg_Creative_Arts_&_Media': 'Creative Arts/Media',
            'JOBctg_Engineering_&_Technology': 'Engineering/Technology',
            'JOBctg_Education_&_Teaching': 'Education/Teaching',
            'JOBctg_Science_&_Research': 'Science/Research',
            'JOBctg_Legal_&_Government': 'Legal/Government',
            'JOBctg_Finance_&_Accounting': 'Finance/Accounting',
            'JOBctg_Social_Services_&_Community': 'Social Services',
            'JOBctg_Construction_&_Surveying': 'Construction/Surveying',
            'JOBctg_Environment_&_Agriculture': 'Environment/Agriculture',
            'JOBctg_Transportation_&_Entertainment_Services': 'Transportation/Entertainment',
            
            # Time descriptions
            'txn_time': 'Transaction Time',
            'avg_txn_time': 'User\'s Typical Transaction Time',
            
            # Amount descriptions
            'amt': 'Transaction Amount',
            'avg_txn_amt': 'User\'s Average Transaction Amount',
            'stddev_txn_amt': 'Amount Variability',
            
            # Distance
            'avg_merchant_distance': 'Distance from User\'s Typical Location'
        }
    
    def _get_transaction_category(self, transaction):
        """Extract the transaction category from one-hot encoded columns"""
        txn_columns = [col for col in transaction.index if col.startswith('TXNctg_')]
        for col in txn_columns:
            if transaction[col] == 1:
                return self.feature_map.get(col, col.replace('TXNctg_', ''))
        return "Unknown Category"
    
    def _get_job_category(self, transaction):
        """Extract the job category from one-hot encoded columns"""
        job_columns = [col for col in transaction.index if col.startswith('JOBctg_')]
        for col in job_columns:
            if transaction[col] == 1:
                return self.feature_map.get(col, col.replace('JOBctg_', ''))
        return "Unknown Profession"
    
    def _get_age_group(self, transaction):
        """Extract the age group from one-hot encoded columns"""
        dob_columns = [col for col in transaction.index if col.startswith('dob_')]
        for col in dob_columns:
            if transaction[col] == 1:
                decade = col.replace('dob_', '').replace('s', '0s')
                return f"born in the {decade}"
        return "unknown age group"
    
    def _format_time(self, hour):
        """Convert hour to human-readable time"""
        if 0 <= hour < 6:
            return f"{int(hour)}:00 AM (early morning)"
        elif 6 <= hour < 12:
            return f"{int(hour)}:00 AM"
        elif 12 <= hour < 18:
            return f"{int(hour-12)}:00 PM" if hour > 12 else "12:00 PM"
        else:
            return f"{int(hour-12)}:00 PM" if hour > 12 else "12:00 PM"
    
    def _extract_transaction_context(self, transaction):
        """Extract human-readable context from transaction features"""
        context = {
            'amount': transaction.get('amt', transaction.get('transaction_amount', 0)),
            'time': transaction.get('txn_time', transaction.get('transaction_hour', 12)),
            'category': self._get_transaction_category(transaction),
            'job': self._get_job_category(transaction),
            'age_group': self._get_age_group(transaction),
            'user_avg_amount': transaction.get('avg_txn_amt', 0),
            'user_avg_time': transaction.get('avg_txn_time', 12),
            'user_avg_distance': transaction.get('avg_merchant_distance', 0),
            'fraud_probability': transaction.get('fraud_probability', 0)
        }
        
        # Calculate anomalies
        amount_ratio = context['amount'] / context['user_avg_amount'] if context['user_avg_amount'] > 0 else 1
        time_difference = abs(context['time'] - context['user_avg_time'])
        
        context['amount_anomaly'] = f"{amount_ratio:.1f}x" if amount_ratio > 1 else "normal"
        context['time_anomaly'] = "significant" if time_difference > 6 else "moderate" if time_difference > 3 else "minor"
        
        return context
    
    def generate_fraud_explanation(self, transaction):
        """Generate LLM explanation for why a transaction is fraudulent"""
        
        # Extract human-readable context
        context = self._extract_transaction_context(transaction)
        
        # Build the user prompt
        user_prompt = f"""
        Transaction Details:
        - Amount: ${context['amount']:.2f} (user's average: ${context['user_avg_amount']:.2f})
        - Time: {self._format_time(context['time'])} (user's typical: {self._format_time(context['user_avg_time'])})
        - Category: {context['category']}
        - User Profile: {context['job']} professional, {context['age_group']}
        - Fraud Confidence: {context['fraud_probability']:.1%}
        
        Key Anomalies:
        - Amount is {context['amount_anomaly']} higher than user's typical spending
        - Time difference shows {context['time_anomaly']} deviation from normal patterns
        
        Explain why this transaction was flagged as fraudulent:
        """
        
        try:
            # Generate explanation using LLM with direct messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Use invoke directly with messages
            response = self.llm.invoke(messages)
            explanation = response.content.strip()
            
            # Clean up the response
            explanation = re.sub(r'^(Explanation|Analysis|Flagged as fraud):?\s*', '', explanation, flags=re.IGNORECASE)
            
            return explanation
            
        except Exception as e:
            print(f"⚠️ LLM Error: {e}. Using fallback explanation.")
            # Fallback explanation if LLM fails
            return f"This ${context['amount']:.2f} transaction was flagged due to unusual patterns: amount is {context['amount_anomaly']} higher than typical spending and occurred at an unusual time ({self._format_time(context['time'])} vs user's normal {self._format_time(context['user_avg_time'])})."
    
    def batch_explain_fraud_transactions(self, fraud_df, max_transactions=50):
        """Generate explanations for multiple fraud transactions"""
        print(f"🧠 Generating LLM explanations for {min(len(fraud_df), max_transactions)} fraud transactions...")
        
        explanations = []
        
        for idx, (_, transaction) in enumerate(fraud_df.head(max_transactions).iterrows()):
            if idx % 10 == 0:
                print(f"   Processed {idx}/{min(len(fraud_df), max_transactions)} transactions...")
            
            explanation = self.generate_fraud_explanation(transaction)
            
            explanations.append({
                'transaction_id': idx,
                'amount': transaction.get('amt', transaction.get('transaction_amount', 0)),
                'time': transaction.get('txn_time', transaction.get('transaction_hour', 12)),
                'category': self._get_transaction_category(transaction),
                'fraud_probability': transaction.get('fraud_probability', 0),
                'explanation': explanation
            })
        
        print("✅ LLM explanations generated successfully!")
        return explanations

# =============================================
# 🚀 MAIN EXECUTION - STANDALONE USAGE
# =============================================

def main():
    """Main function for standalone testing"""
    
    # Check if required files exist
    if not os.path.exists('detected_fraud_transactions.csv'):
        print("❌ 'detected_fraud_transactions.csv' not found. Run inference first.")
        return
    
    # Load fraud transactions
    fraud_df = pd.read_csv('detected_fraud_transactions.csv')
    print(f"📊 Loaded {len(fraud_df)} fraud transactions")
    
    # Initialize explainer
    explainer = FraudExplainer()
    
    # Generate explanations (limit to first 20 for testing)
    explanations = explainer.batch_explain_fraud_transactions(fraud_df, max_transactions=20)
    
    # Save explanations
    output_data = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'total_transactions': len(explanations),
        'explanations': explanations
    }
    
    with open('llm_fraud_explanations.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Saved {len(explanations)} LLM explanations to 'llm_fraud_explanations.json'")
    
    # Display sample explanations
    print("\n📝 SAMPLE EXPLANATIONS:")
    print("=" * 50)
    for i, exp in enumerate(explanations[:3]):
        print(f"\n🚨 Fraud #{i+1}:")
        print(f"   Amount: ${exp['amount']:.2f} | Time: {explainer._format_time(exp['time'])}")
        print(f"   Category: {exp['category']} | Confidence: {exp['fraud_probability']:.1%}")
        print(f"   Explanation: {exp['explanation']}")

if __name__ == "__main__":
    main()