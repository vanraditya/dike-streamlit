import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib

# Load model and scaler
MODEL_PATH = 'models/mlp_model_final.pt'
SCALER_PATH = 'models/scaler.pkl'

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model_and_scaler():
    model = MLP(input_dim=5)  # ✅ Correct class and input_dim
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    scaler = joblib.load(SCALER_PATH)

    return model, scaler


model, scaler = load_model_and_scaler()

# Streamlit app UI
st.title("🧠 Malware Detection using MLP (5 Features)")

st.markdown("### Enter feature values below:")

feature_names = ['spyware', 'encrypter', 'downloader', 'backdoor', 'ransomware']
input_data = []

for feature in feature_names:
    value = st.number_input(f"{feature}:", min_value=0.0, max_value=1.0, step=0.01)
    input_data.append(value)

if st.button("Predict"):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output >= 0.5).item()

    if prediction == 1:
        st.error("🔴 This file is predicted to be **Malware**.")
    else:
        st.success("🟢 This file is predicted to be **Benign**.")
