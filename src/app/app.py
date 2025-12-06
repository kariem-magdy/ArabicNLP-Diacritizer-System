# src/app/app.py
from flask import Flask, request, jsonify, render_template_string
from ..infer.infer import DiacriticPredictor
from ..infer.infer_transformer import TransformerPredictor
import sys

app = Flask(__name__)

# --- Initialize Models ---
print("------------------------------------------------")
print("[INFO] Loading BiLSTM-CRF Model...")
try:
    bilstm_predictor = DiacriticPredictor()
    print("[SUCCESS] BiLSTM Loaded.")
except Exception as e:
    print(f"[ERROR] Could not load BiLSTM: {e}")
    bilstm_predictor = None

print("\n[INFO] Loading Transformer (AraBERT) Model...")
try:
    trans_predictor = TransformerPredictor()
    print("[SUCCESS] Transformer Loaded.")
except Exception as e:
    print(f"[ERROR] Could not load Transformer: {e}")
    trans_predictor = None
print("------------------------------------------------")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <title>Arabic Diacritizer</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; padding: 20px; }
        .container { max-width: 700px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h2 { color: #2c3e50; text-align: center; }
        textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 1.2em; font-family: 'Courier New', Courier, monospace; resize: vertical; }
        select { padding: 10px; width: 100%; margin-bottom: 20px; font-size: 1em; border-radius: 5px; border: 1px solid #ddd; }
        button { background-color: #3498db; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 1.1em; width: 100%; }
        button:hover { background-color: #2980b9; }
        #output_text { margin-top: 20px; font-size: 1.4em; padding: 15px; border: 1px solid #eee; background: #fafafa; border-radius: 5px; min-height: 60px; line-height: 1.8; }
        .label { font-weight: bold; margin-bottom: 5px; display: block; color: #555; }
    </style>
</head>
<body>
    <div class="container">
        <h2>نظام تشكيل النصوص العربية</h2>
        
        <label class="label">اختر النموذج (Choose Model):</label>
        <select id="model_select">
            <option value="bilstm">BiLSTM-CRF (Baseline)</option>
            <option value="transformer">Transformer (AraBERT)</option>
        </select>

        <label class="label">النص (Input Text):</label>
        <textarea id="input_text" rows="5" placeholder="أدخل النص العربي غير المشكول هنا..."></textarea>
        <br><br>
        
        <button onclick="diacritize()">تشكيل (Diacritize)</button>
        
        <label class="label" style="margin-top: 20px;">النتيجة (Result):</label>
        <p id="output_text"></p>
    </div>

    <script>
        async function diacritize() {
            const text = document.getElementById('input_text').value;
            const model = document.getElementById('model_select').value;
            const outBox = document.getElementById('output_text');
            
            outBox.style.color = "#999";
            outBox.textContent = "جاري المعالجة...";

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({text: text, model: model})
                });
                const data = await response.json();
                
                if (data.error) {
                    outBox.style.color = "red";
                    outBox.textContent = "Error: " + data.error;
                } else {
                    outBox.style.color = "#000";
                    outBox.textContent = data.diacritized;
                }
            } catch (err) {
                outBox.style.color = "red";
                outBox.textContent = "Connection Error";
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    model_type = data.get('model', 'bilstm')

    if not text:
        return jsonify({'diacritized': ''})
    
    try:
        if model_type == 'transformer':
            if trans_predictor:
                result = trans_predictor.predict(text)
            else:
                return jsonify({'error': 'Transformer model is not loaded correctly.'})
        else:
            if bilstm_predictor:
                result = bilstm_predictor.predict(text)
            else:
                return jsonify({'error': 'BiLSTM model is not loaded correctly.'})
                
        return jsonify({'diacritized': result})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)