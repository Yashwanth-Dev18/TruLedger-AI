from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import json
import os
from dataProcessor import process_transaction_data
from ML_Engine import run_fraud_detection

app = Flask(__name__)

# =============================================
# 📊 DATA LOADING FUNCTIONS
# =============================================

def load_fraud_data():
    try:
        return pd.read_csv('detected_fraud_transactions.csv')
    except FileNotFoundError:
        return None

def load_visualization_data():
    try:
        with open('app_visualization_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# =============================================
# 📈 HTML GENERATORS (copied from Gradio version)
# =============================================

def display_metrics():
    return """
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 20px 0;">
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #1f77b4;">Precision</h3><p><b>0.73</b></p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #1f77b4;">Recall</h3><p><b>0.89</b></p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #1f77b4;">F1-Score</h3><p><b>0.80</b></p>
        </div>
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #1f77b4;">False Positives</h3><p><b>492</b></p>
        </div>
    </div>
    """

def display_chart(data, title, chart_type="bar"):
    if not data:
        return f"<p>No data for {title}</p>"
    
    max_val = max(data.values()) if data.values() else 1
    bars = ""
    for category, value in data.items():
        pct = (value / max_val) * 100
        bars += f"""
        <div style="margin-bottom:10px">
            <strong>{category}</strong>
            <div style="background:#e0e0e0; border-radius:5px;">
                <div style="background:#1f77b4; width:{pct}%; height:20px; border-radius:5px;"></div>
            </div>
        </div>"""
    return f"<div><h4>{title}</h4>{bars}</div>"

# =============================================
# 🚀 PIPELINE FUNCTION
# =============================================

def run_fraud_pipeline(selected_dataset, groq_api_key=None):
    if groq_api_key:
        os.environ['GROQ_API_KEY'] = groq_api_key

    dataset_files = {
        "TransactionLogs-1": "TransactionLogs-1.csv",
        "TransactionLogs-2": "TransactionLogs-2.csv",
        "TransactionLogs-3": "TransactionLogs-3.csv"
    }

    results = {"status": "success", "message": "", "html": ""}

    try:
        input_file = dataset_files.get(selected_dataset)
        if not input_file:
            return {"status": "error", "message": "Invalid dataset selected."}

        output_file = f"Processed_{selected_dataset}.csv"
        processed = process_transaction_data(input_file, output_file)
        if not processed:
            return {"status": "error", "message": "Data processing failed."}

        success = run_fraud_detection(processed)
        if not success:
            return {"status": "error", "message": "Fraud detection failed."}

        fraud_df = load_fraud_data()
        viz_data = load_visualization_data()
        if fraud_df is None or viz_data is None:
            return {"status": "error", "message": "Could not load results."}

        charts_html = ""
        if 'job_categories' in viz_data:
            charts_html += display_chart(viz_data['job_categories'], "Top Job Categories")
        if 'age_groups' in viz_data:
            charts_html += display_chart(viz_data['age_groups'], "Age Groups")
        if 'transaction_categories' in viz_data:
            charts_html += display_chart(viz_data['transaction_categories'], "Transaction Categories")

        results["html"] = f"""
        <h2>✅ Pipeline Complete</h2>
        <p>Found <b>{len(fraud_df)}</b> suspicious transactions.</p>
        {display_metrics()}
        {charts_html}
        <h3>Detected Fraud (Top 10)</h3>
        {fraud_df.head(10).to_html(classes='dataframe', index=False)}
        """

    except Exception as e:
        results["status"] = "error"
        results["message"] = str(e)

    return results

# =============================================
# 🌍 ROUTES
# =============================================

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head><title>TruLedger - Fraud Detection</title></head>
    <body style="font-family:sans-serif; padding:20px;">
        <h1>🏦 TruLedger - AI Fraud Detection</h1>
        <form method="post" action="/run">
            <label>Select Dataset:</label>
            <select name="dataset">
                <option>TransactionLogs-1</option>
                <option>TransactionLogs-2</option>
                <option>TransactionLogs-3</option>
            </select><br><br>
            <label>Groq API Key (optional):</label><br>
            <input type="password" name="api_key"><br><br>
            <button type="submit">🚀 Run Fraud Detection</button>
        </form>
    </body>
    </html>
    """)

@app.route('/run', methods=['POST'])
def run_pipeline():
    dataset = request.form.get('dataset')
    api_key = request.form.get('api_key')
    results = run_fraud_pipeline(dataset, api_key)

    if results["status"] == "error":
        return f"<h2 style='color:red'>Error: {results['message']}</h2>", 400
    return render_template_string(results["html"])

# =============================================
# 🏁 ENTRY POINT
# =============================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
