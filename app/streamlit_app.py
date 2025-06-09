import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pickle

# Load model and scaler
MODEL_PATH = 'models/mlp_model_final.pt'
SCALER_PATH = 'models/scaler.pkl'

class MLPClassifier(nn.Module):
    def __init__(self, input_size=5):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_model_and_scaler():
    model = MLPClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

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
