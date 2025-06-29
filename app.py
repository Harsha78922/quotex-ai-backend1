from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load your trained model
try:
    model = tf.keras.models.load_model("model.h5")
except Exception as e:
    model = None
    print("Model load failed:", e)

def preprocess_input(data):
    try:
        sequence = [[
            candle['open'], candle['high'],
            candle['low'], candle['close'], candle['volume']
        ] for candle in data]
        return np.array([sequence])
    except Exception as e:
        print("Preprocessing failed:", e)
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    content = request.get_json()
    candles = content.get("candles")

    input_data = preprocess_input(candles)
    if input_data is None:
        return jsonify({"error": "Invalid input format"}), 400

    prediction = model.predict(input_data)[0]
    signal = "Buy" if prediction[0] > 0.5 else "Sell"
    confidence = int(prediction[0] * 100) if signal == "Buy" else int((1 - prediction[0]) * 100)

    return jsonify({
        "signal": signal,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)