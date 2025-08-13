from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Загружаем пайплайн
with open("pipeline.pkl", "rb") as f:
    pipe = pickle.load(f)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # ожидаем dict
        df = pd.DataFrame([data])
        prediction = pipe.predict(df)[0]
        return jsonify({"prediction": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run("0.0.0.0", 5000)
