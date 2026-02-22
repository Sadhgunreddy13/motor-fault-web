from flask import Flask, render_template, request
import joblib
import torch
import librosa
import numpy as np
from transformers import WavLMModel

app = Flask(__name__)

# Load trained classifier
model = joblib.load("model/HistGB.pkl")
le = joblib.load("model/label_encoder.pkl")

# Load WavLM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
wavlm_model.eval()

def extract_feature(audio):
    inputs = torch.tensor(audio).float().unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = wavlm_model(inputs)
        feat = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return feat

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["audio"]

    y, sr = librosa.load(file, sr=16000)
    feat = extract_feature(y)

    prediction_num = model.predict(feat)[0]
    prediction_label =le.inverse_transform([prediction_num])[0]

    return render_template("index.html", result=prediction_label)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

