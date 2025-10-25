import gradio as gr
import pandas as pd
import numpy as np
import json
import time
import os
import sys
from datetime import datetime

# Import our processing functions
from dataProcessor import process_transaction_data
from app_ML_Engine import run_fraud_detection

# =============================================
# 📊 DATA LOADING FUNCTIONS
# =============================================

def load_fraud_data():
    """Load detected fraud transactions"""
    try:
        fraud_df = pd.read_csv('detected_fraud_transactions.csv')
        return fraud_df
    except FileNotFoundError:
        return None

def load_visualization_data():
    """Load pre-computed visualization data"""
    try:
        with open('app_visualization_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# =============================================
# 🎯 GRADIO COMPONENT FUNCTIONS
# =============================================

def display_metrics():
    """Display model performance metrics"""
    metrics_html = """
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0;">
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;">
            <h3 style="margin: 0; color: #1f77b4;">Precision</h3>
            <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">0.73</p>
            <p style="color: green; margin: 0;">+0.34</p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;">
            <h3 style="margin: 0; color: #1f77b4;">Recall</h3>
            <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">0.89</p>
            <p style="color: green; margin: 0;">+0.07</p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;">
            <h3 style="margin: 0; color: #1f77b4;">F1-Score</h3>
            <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">0.80</p>
            <p style="color: green; margin: 0;">+0.24</p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center; border: 1px solid #e0e0e0;">
            <h3 style="margin: 0; color: #1f77b4;">False Positives</h3>
            <p style="font-size: 24px; font-weight: bold; margin: 10px 0;">492</p>
            <p style="color: red; margin: 0;">-77.9%</p>
        </div>
    </div>
    """
    return metrics_html

def display_model_comparison():
    """Display model comparison table"""
    comparison_data = pd.DataFrame({
        'Model': ['XGBoost', 'Autoencoder', 'Isolation Forest'],
        'Precision': [0.73, 0.06, 0.07],
        'Recall': [0.89, 0.11, 0.24],
        'F1-Score': [0.80, 0.08, 0.11]
    })
    
    # Convert to HTML table with styling
    html_table = comparison_data.to_html(
        index=False, 
        classes='dataframe', 
        float_format='%.2f',
        border=0
    )
    
    # Add custom styling
    styled_html = f"""
    <div style="margin: 20px 0;">
        <h3>Model Comparison</h3>
        <style>
            .dataframe {{
                width: 100%;
                border-collapse: collapse;
            }}
            .dataframe th {{
                background-color: #1f77b4;
                color: white;
                padding: 10px;
                text-align: left;
            }}
            .dataframe td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            .dataframe tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            .dataframe tr:hover {{
                background-color: #e1f5fe;
            }}
        </style>
        {html_table}
    </div>
    """
    return styled_html

def display_chart(data, title, chart_type="bar"):
    """Display chart data as HTML"""
    if not data:
        return f"<p>No data available for {title}</p>"
    
    # Create a simple HTML representation of the chart
    items_html = []
    max_val = max(data.values()) if data.values() else 1
    
    for category, value in data.items():
        clean_category = category.replace('JOBctg_', '').replace('_', ' ')
        percentage = (value / max_val) * 100
        
        if chart_type == "bar":
            items_html.append(f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span><strong>{clean_category}</strong></span>
                    <span>{value}</span>
                </div>
                <div style="background: #e0e0e0; border-radius: 5px; height: 20px;">
                    <div style="background: #1f77b4; height: 100%; border-radius: 5px; width: {percentage}%;"></div>
                </div>
            </div>
            """)
        elif chart_type == "pie":
            items_html.append(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 5px 0;">
                <span><strong>{clean_category}</strong></span>
                <span>{value} ({percentage:.1f}%)</span>
            </div>
            """)
    
    return f"""
    <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h4>{title}</h4>
        {"".join(items_html)}
    </div>
    """

# =============================================
# 🚀 MAIN PROCESSING FUNCTION
# =============================================

def run_fraud_pipeline(selected_dataset, groq_api_key=None):
    """Main function to run the fraud detection pipeline"""
    
    # Set API key if provided
    if groq_api_key:
        os.environ['GROQ_API_KEY'] = groq_api_key
    
    # Dataset mapping
    dataset_files = {
        "TransactionLogs-1": "TransactionLogs-1.csv",
        "TransactionLogs-2": "TransactionLogs-2.csv", 
        "TransactionLogs-3": "TransactionLogs-3.csv"
    }
    
    results = {
        "status": "success",
        "message": "",
        "fraud_count": 0,
        "metrics_html": "",
        "model_comparison_html": "",
        "charts_html": "",
        "fraud_table_html": ""
    }
    
    try:
        # Process data
        input_file = dataset_files.get(selected_dataset)
        if not input_file:
            results["status"] = "error"
            results["message"] = "❌ Invalid dataset selected"
            return results
        
        output_file = f"Processed_{selected_dataset}.csv"
        
        # Process transaction data
        processed_file = process_transaction_data(input_file, output_file)
        if processed_file is None:
            results["status"] = "error"
            results["message"] = "❌ Failed to process data"
            return results
        
        # Run fraud detection
        success = run_fraud_detection(processed_file)
        if not success:
            results["status"] = "error"
            results["message"] = "❌ Failed to run fraud detection"
            return results
        
        # Load results
        fraud_df = load_fraud_data()
        viz_data = load_visualization_data()
        
        if fraud_df is not None and viz_data is not None:
            fraud_count = len(fraud_df)
            results.update({
                "fraud_count": fraud_count,
                "message": f"✅ Pipeline complete! Found {fraud_count} suspicious transactions.",
                "metrics_html": display_metrics(),
                "model_comparison_html": display_model_comparison(),
                "fraud_table_html": fraud_df.head(10).to_html(classes='dataframe', index=False)
            })
            
            # Generate charts HTML
            charts_html = "<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>"
            
            if 'job_categories' in viz_data:
                charts_html += display_chart(viz_data['job_categories'], "👔 Top Job Categories", "bar")
            
            if 'age_groups' in viz_data:
                charts_html += display_chart(viz_data['age_groups'], "👥 Age Groups", "pie")
            
            if 'transaction_categories' in viz_data:
                charts_html += display_chart(viz_data['transaction_categories'], "🛒 Transaction Categories", "bar")
            
            if 'amount_analysis' in viz_data:
                amt_data = viz_data['amount_analysis']
                amount_html = f"""
                <div style="background: #f8f9fa; padding: 20px; border-radius: 10px;">
                    <h4>💰 Amount Analysis</h4>
                    <p><strong>Normal Avg Amount:</strong> ${amt_data['normal_avg']:.2f}</p>
                    <p><strong>Fraud Avg Amount:</strong> ${amt_data['fraud_avg']:.2f}</p>
                    <p><strong>Increase:</strong> +{amt_data['increase_pct']:.1f}%</p>
                </div>
                """
                charts_html += amount_html
            
            charts_html += "</div>"
            results["charts_html"] = charts_html
            
        else:
            results["status"] = "error"
            results["message"] = "❌ Failed to load analysis results"
            
    except Exception as e:
        results["status"] = "error"
        results["message"] = f"❌ Error: {str(e)}"
    
    return results

# =============================================
# 🎨 GRADIO INTERFACE
# =============================================

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="TruLedger - AI Fraud Detection",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .header {
            text-align: center;
            padding: 20px;
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
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1 style="font-size: 3rem; color: #1f77b4; margin-bottom: 0;">🏦 TruLedger</h1>
            <p style="font-size: 1.2rem; color: #666;">
                An Explainable AI Prototype for ML-powered Fraud Detection in Finance Records
            </p>
        </div>
        """)
        
        # Technologies Section
        gr.HTML("""
        <div style="text-align: center; margin: 20px 0;">
            <span class="tech-badge">PySpark</span>
            <span class="tech-badge">XGBoost</span>
            <span class="tech-badge">SHAP</span>
            <span class="tech-badge">LLM (Groq)</span>
            <span class="tech-badge">TensorFlow</span>
            <span class="tech-badge">Feature Engineering</span>
            <span class="tech-badge">Anomaly Detection</span>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Dataset Selection
                dataset_dropdown = gr.Dropdown(
                    choices=[
                        "TransactionLogs-1",
                        "TransactionLogs-2", 
                        "TransactionLogs-3"
                    ],
                    label="📁 Choose Dataset to Analyze",
                    value="TransactionLogs-1",
                    info="Select a dataset to run the complete fraud detection pipeline"
                )
                
                # API Key Input
                api_key_input = gr.Textbox(
                    label="🔑 Groq API Key (Optional)",
                    type="password",
                    placeholder="Enter your Groq API key for AI explanations...",
                    info="Get your API key from https://console.groq.com"
                )
                
                # Run Button
                run_button = gr.Button(
                    "🚀 Run Complete Fraud Detection Pipeline",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                # Results Output
                status_output = gr.HTML(
                    label="Status",
                    value="<div style='padding: 20px; text-align: center; color: #666;'>Ready to analyze financial transactions...</div>"
                )
                
                # Metrics Output
                metrics_output = gr.HTML(label="Model Metrics")
                
                # Charts Output
                charts_output = gr.HTML(label="Fraud Analysis")
                
                # Model Comparison
                model_output = gr.HTML(label="Model Comparison")
                
                # Fraud Transactions Table
                fraud_table_output = gr.HTML(label="Detected Fraud Transactions")
        
        # Footer
        gr.HTML("""
        <div style='text-align: center; color: #666; margin-top: 40px; padding: 20px; border-top: 1px solid #e0e0e0;'>
            <p>🔒 <strong>TruLedger</strong> - Explainable AI for Financial Security</p>
            <p>Built with PySpark, XGBoost, SHAP, and LLM technologies</p>
        </div>
        """)
        
        # Event Handling
        run_button.click(
            fn=run_fraud_pipeline,
            inputs=[dataset_dropdown, api_key_input],
            outputs=[
                status_output,
                metrics_output, 
                charts_output,
                model_output,
                fraud_table_output
            ]
        )
    
    return demo

# =============================================
# 🚀 APP INITIALIZATION
# =============================================

if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_interface()
    demo.launch(
        share=False,  # Set to True if you want a public link
        debug=True,   # Set to False in production
        show_error=True
    )