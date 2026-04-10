import os
import io
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from collections import Counter

# Import your AgriculturalAI class here or ensure it's in the same file
# I've modified the predict_disease slightly to handle byte streams from Flask
from app import AgriculturalAI, IDX_TO_CLASS 

app = Flask(__name__)
CORS(app) # Enables Cross-Origin Resource Sharing

# Initialize the AI Engine
# Ensure your .h5 and .pkl files are in the directory specified below
ai_system = AgriculturalAI(models_dir="models")

# ==========================================
# 1. Disease Detection Endpoint (Image)
# ==========================================
@app.route('/predict-disease', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read()))
        # Ensure image is RGB (removes alpha channel if exists)
        img = img.convert('RGB')
        
        result = ai_system.predict_disease(img)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 2. Soil -> Crop & Fertilizer Endpoint
# ==========================================
@app.route('/recommend-all', methods=['POST'])
def recommend_all():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    try:
        # Expected keys: Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, Soil_color
        result = ai_system.recommend_crop_and_fert(data)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# 3. Soil + Crop -> Fertilizer Endpoint
# ==========================================
@app.route('/recommend-fertilizer', methods=['POST'])
def recommend_fertilizer():
    data = request.json
    try:
        # Expected keys: Nitrogen, Phosphorus, Potassium, pH, Rainfall, Temperature, Soil_color, Crop
        df = pd.DataFrame([data])
        
        fert_pred = ai_system.models['fert_model'].predict(df)
        fert_name = ai_system.models['le_fert'].inverse_transform(fert_pred)[0]
        
        return jsonify({"recommended_fertilizer": fert_name}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on port 5000
    app.run(debug=True, port=5000)
    # app.run(host='0.0.0.0', port=5000)