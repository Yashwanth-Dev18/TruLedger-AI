import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import sys
try:
    from app_LLM_XAI import FraudExplainer
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ LLM module not available: {e}")
    LLM_AVAILABLE = False

# Import our processing functions
from dataProcessor import process_transaction_data
from app_ML_Engine import run_fraud_detection

# =============================================
# 🎯 PAGE CONFIGURATION
# =============================================

st.set_page_config(
    page_title="TruLedger - AI Fraud Detection",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# 🎨 CUSTOM CSS STYLING
# =============================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .tech-badge {
        background-color: #e1f5fe;
        padding: 8px 16px;
        border-radius: 20px;
        margin: 5px;
        display: inline-block;
        font-weight: 500;
    }
    .fraud-card {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .processing-step {
        background-color: #e8f5e8;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid #4caf50;
    }
    .chart-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# 📊 DATA LOADING FUNCTIONS
# =============================================

@st.cache_data
def load_fraud_data():
    """Load detected fraud transactions"""
    try:
        fraud_df = pd.read_csv('detected_fraud_transactions.csv')
        return fraud_df
    except FileNotFoundError:
        return None

@st.cache_data
def load_visualization_data():
    """Load pre-computed visualization data"""
    try:
        with open('app_visualization_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_llm_explanations():
    """Load LLM explanations"""
    try:
        with open('llm_fraud_explanations.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# =============================================
# 📊 STREAMLIT CHART FUNCTIONS (No Plotly)
# =============================================

def display_bar_chart(data, title, x_label="Count", y_label="Category"):
    """Display a bar chart using Streamlit elements"""
    if not data:
        st.info("No data available for chart")
        return
    
    st.subheader(title)
    
    # Convert to DataFrame for display
    chart_data = pd.DataFrame({
        'Category': [key.replace('JOBctg_', '').replace('_', ' ') for key in data.keys()],
        'Count': list(data.values())
    })
    
    # Display as bar chart using st.bar_chart
    st.bar_chart(chart_data.set_index('Category'))
    
    # Also show as table for clarity
    st.dataframe(chart_data, use_container_width=True)

def display_pie_chart(data, title):
    """Display a pie chart using Streamlit progress bars"""
    if not data:
        st.info("No data available for chart")
        return
    
    st.subheader(title)
    
    total = sum(data.values())
    for category, count in data.items():
        percentage = (count / total) * 100
        clean_category = category.replace('dob_', '').upper()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{clean_category}**")
        with col2:
            st.progress(percentage/100, text=f"{percentage:.1f}%")
        with col3:
            st.write(f"{count} cases")

def display_horizontal_bar_chart(data, title):
    """Display horizontal bar chart using Streamlit"""
    if not data:
        st.info("No data available for chart")
        return
    
    st.subheader(title)
    
    for category, count in data.items():
        clean_category = category.replace('TXNctg_', '').replace('_', ' ')
        max_val = max(data.values())
        progress = count / max_val
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{clean_category}**")
        with col2:
            st.progress(progress, text=f"{count} cases")

# =============================================
# 🎯 MAIN APP FUNCTION
# =============================================

def main():
    # =============================================
    # 🏦 HEADER SECTION
    # =============================================
    
    st.markdown('<h1 class="main-header">🏦 TruLedger</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">An Explainable AI Prototype for ML-powered Fraud Detection in Finance Records</p>', unsafe_allow_html=True)
    
    # =============================================
    # 🔧 TECHNOLOGIES SECTION
    # =============================================
    
    st.markdown("---")
    st.header("🛠️ Technologies Involved")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="tech-badge">PySpark</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">XGBoost</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="tech-badge">SHAP</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">LLM (Groq)</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="tech-badge">Streamlit</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">Data Visualization</div>', unsafe_allow_html=True)
        
    with col4:
        st.markdown('<div class="tech-badge">Feature Engineering</div>', unsafe_allow_html=True)
        st.markdown('<div class="tech-badge">Anomaly Detection</div>', unsafe_allow_html=True)
    
    # =============================================
    # 🔑 API KEY SETUP SECTION
    # =============================================
    
    st.markdown("---")
    st.header("🔑 Groq API Setup")
    
    # Check if API key exists
    if os.getenv('GROQ_API_KEY'):
        st.success("✅ GROQ_API_KEY found in environment variables")
    else:
        st.warning("⚠️ GROQ_API_KEY not found in environment variables")
        api_key = st.text_input(
            "Enter your Groq API Key:",
            type="password",
            help="Get your API key from https://console.groq.com"
        )
        
        if api_key:
            os.environ['GROQ_API_KEY'] = api_key
            st.success("✅ API key set! You can now generate AI explanations.")
    
    # =============================================
    # 📁 DATASET SELECTION & PROCESSING SECTION
    # =============================================
    
    st.markdown("---")
    st.header("📁 Dataset Selection & Analysis")
    
    # Dataset options
    dataset_options = {
        "TransactionLogs-1": "Small business transactions (1,000 records)",
        "TransactionLogs-2": "Medium enterprise transactions (5,000 records)", 
        "TransactionLogs-3": "Large financial logs (10,000 records)"
    }
    
    selected_dataset = st.selectbox(
        "Choose a dataset to analyze:",
        options=list(dataset_options.keys()),
        help="Select a dataset to run the complete fraud detection pipeline"
    )
    
    st.info(f"**{selected_dataset}**: {dataset_options[selected_dataset]}")
    
    if st.button("🚀 Run Complete Fraud Detection Pipeline", type="primary"):
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Data Processing
        status_text.markdown('<div class="processing-step">🔄 Step 1/3: Processing raw data...</div>', unsafe_allow_html=True)
        progress_bar.progress(25)
        
        input_file = f"c:/Users/hp/LNU/TruLedger-AI/{selected_dataset}.csv"
        output_file = f"Processed{selected_dataset}.csv"
        
        processed_file = process_transaction_data(input_file, output_file)
        
        if processed_file is None:
            st.error("❌ Failed to process data. Please check the file path and try again.")
            return
            
        # Step 2: Fraud Detection
        status_text.markdown('<div class="processing-step">🔍 Step 2/3: Running fraud detection...</div>', unsafe_allow_html=True)
        progress_bar.progress(60)
        
        success = run_fraud_detection(processed_file)
        
        if not success:
            st.error("❌ Failed to run fraud detection. Please check the model files.")
            return
            
        # Step 3: Load Results
        status_text.markdown('<div class="processing-step">📊 Step 3/3: Loading results...</div>', unsafe_allow_html=True)
        progress_bar.progress(90)
        
        fraud_df = load_fraud_data()
        viz_data = load_visualization_data()
        
        if fraud_df is not None and viz_data is not None:
            progress_bar.progress(100)
            status_text.markdown('<div class="processing-step">✅ Analysis complete!</div>', unsafe_allow_html=True)
            
            st.success(f"✅ Pipeline complete! Found {len(fraud_df)} suspicious transactions.")
            st.session_state.analysis_complete = True
            st.session_state.fraud_df = fraud_df
            st.session_state.viz_data = viz_data
            st.session_state.selected_dataset = selected_dataset
        else:
            st.error("❌ Failed to load analysis results.")
            st.session_state.analysis_complete = False
    
    # =============================================
    # 📊 ML MODEL STATS SECTION
    # =============================================
    
    if st.session_state.get('analysis_complete', False):
        st.markdown("---")
        st.header("📊 ML Model Performance")
        
        # Show which dataset is being analyzed
        if st.session_state.get('selected_dataset'):
            st.info(f"📁 Currently analyzing: **{st.session_state.selected_dataset}**")
        
        # Model metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precision", "0.73", "0.34")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Recall", "0.89", "0.07")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("F1-Score", "0.80", "0.24")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("False Positives", "492", "-77.9%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Model comparison table
        st.subheader("Model Comparison")
        
        comparison_data = pd.DataFrame({
            'Model': ['XGBoost', 'Autoencoder', 'Isolation Forest'],
            'Precision': [0.73, 0.06, 0.07],
            'Recall': [0.89, 0.11, 0.24],
            'F1-Score': [0.80, 0.08, 0.11]
        })
        
        st.dataframe(
            comparison_data.style.format({
                'Precision': '{:.2f}',
                'Recall': '{:.2f}', 
                'F1-Score': '{:.2f}'
            }).highlight_max(color='lightgreen'),
            use_container_width=True
        )
    
    # =============================================
    # 📈 ANOMALY DETECTION RESULTS
    # =============================================
    
    if st.session_state.get('analysis_complete', False):
        st.markdown("---")
        st.header("🔍 Fraud Pattern Analysis")
        
        viz_data = st.session_state.viz_data
        
        # Create two columns for charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 5 Job Categories in Fraud
            if 'job_categories' in viz_data:
                display_bar_chart(
                    viz_data['job_categories'],
                    "👔 Top Job Categories Involved in Fraud"
                )
            
            # Amount Analysis
            if 'amount_analysis' in viz_data:
                amt_data = viz_data['amount_analysis']
                st.subheader("💰 Amount Analysis")
                col_a1, col_a2, col_a3 = st.columns(3)
                
                with col_a1:
                    st.metric(
                        "Normal Avg Amount", 
                        f"${amt_data['normal_avg']:.2f}"
                    )
                
                with col_a2:
                    st.metric(
                        "Fraud Avg Amount", 
                        f"${amt_data['fraud_avg']:.2f}"
                    )
                
                with col_a3:
                    st.metric(
                        "Increase", 
                        f"+{amt_data['increase_pct']:.1f}%"
                    )
        
        with col2:
            # Age Groups
            if 'age_groups' in viz_data:
                display_pie_chart(
                    viz_data['age_groups'],
                    "👥 Age Groups in Fraud"
                )
            
            # Top 5 Transaction Categories
            if 'transaction_categories' in viz_data:
                display_horizontal_bar_chart(
                    viz_data['transaction_categories'],
                    "🛒 Top Transaction Categories in Fraud"
                )
    
    # =============================================
    # 🧠 LLM EXPLANATIONS SECTION
    # =============================================
    if st.session_state.get('analysis_complete', False):
        st.markdown("---")
        st.header("🧠 AI Fraud Explanations")
    
    if not LLM_AVAILABLE:
        st.warning("🔧 LLM feature temporarily unavailable - installing dependencies...")
        st.info("The core fraud detection is working! LLM explanations will be available once dependencies are installed.")
    else:
        
        if st.session_state.get('analysis_complete', False):
            st.markdown("---")
            st.header("🧠 AI Fraud Explanations")

            # Load or generate LLM explanations
            llm_data = load_llm_explanations()

            if llm_data is None:
                st.warning("LLM explanations not found. Click below to generate AI explanations.")
                if st.button("Generate AI Explanations"):
                    with st.spinner("Generating AI explanations for fraud cases..."):
                        try:
                            explainer = FraudExplainer()
                            fraud_df = st.session_state.fraud_df
                            explanations = explainer.batch_explain_fraud_transactions(fraud_df, max_transactions=50)

                            llm_data = {
                                'generated_at': pd.Timestamp.now().isoformat(),
                                'total_transactions': len(explanations),
                                'explanations': explanations
                            }

                            with open('llm_fraud_explanations.json', 'w') as f:
                                json.dump(llm_data, f, indent=2)

                            st.session_state.llm_data = llm_data
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error generating explanations: {e}")
            else:
                st.session_state.llm_data = llm_data

            # Display fraud transactions with explanations
            if st.session_state.get('llm_data'):
                explanations = st.session_state.llm_data['explanations']

                # Pagination
                if 'page' not in st.session_state:
                    st.session_state.page = 0

                items_per_page = 6
                total_pages = (len(explanations) + items_per_page - 1) // items_per_page

                # Get current page items
                start_idx = st.session_state.page * items_per_page
                end_idx = min(start_idx + items_per_page, len(explanations))
                current_items = explanations[start_idx:end_idx]

                st.subheader(f"Fraud Cases ({len(explanations)} total)")

                # Display fraud cards
                for i, exp in enumerate(current_items):
                    with st.container():
                        st.markdown(f"""
                        <div class="fraud-card">
                            <h4>🚨 Fraud Alert #{start_idx + i + 1}</h4>
                            <p><strong>Amount:</strong> ${exp['amount']:.2f} | 
                            <strong>Time:</strong> {exp['time']}:00 | 
                            <strong>Category:</strong> {exp['category']} | 
                            <strong>Confidence:</strong> {exp['fraud_probability']:.1%}</p>
                            <p><strong>AI Explanation:</strong> {exp['explanation']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Pagination controls
                col1, col2, col3 = st.columns([1, 2, 1])

                with col1:
                    if st.session_state.page > 0:
                        if st.button("◀ Previous 6"):
                            st.session_state.page -= 1
                            st.rerun()

                with col2:
                    st.write(f"Page {st.session_state.page + 1} of {total_pages}")

                with col3:
                    if st.session_state.page < total_pages - 1:
                        if st.button("Next 6 ▶"):
                            st.session_state.page += 1
                            st.rerun()

                st.info(f"Showing {len(current_items)} of {len(explanations)} fraud cases")

    # =============================================
    # 📝 FOOTER
    # =============================================
    
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🔒 <strong>TruLedger</strong> - Explainable AI for Financial Security</p>
            <p>Built with PySpark, XGBoost, SHAP, and LLM technologies</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================
# 🚀 APP INITIALIZATION
# =============================================

if __name__ == "__main__":
    # Initialize session state
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'page' not in st.session_state:
        st.session_state.page = 0
    if 'selected_dataset' not in st.session_state:
        st.session_state.selected_dataset = None
    
    main()