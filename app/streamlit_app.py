import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
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
    model = MLP(input_dim=5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()
feature_names = ['spyware', 'encrypter', 'downloader', 'backdoor', 'ransomware']

# UI
st.title("🧠 Malware Detection using MLP (5 Features)")
st.markdown("Select input method:")

option = st.radio("Choose input type", ['Manual Entry', 'CSV File Upload'])

if option == 'Manual Entry':
    st.markdown("### Enter feature values below:")
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

else:
    st.markdown("### Upload CSV file for inference")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if all(feature in df.columns for feature in feature_names):
                input_array = df[feature_names].values
                input_scaled = scaler.transform(input_array)
                input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    predictions = (outputs >= 0.5).int().flatten().numpy()

                df['Prediction'] = predictions
                df['Prediction Label'] = df['Prediction'].map({0: 'Benign', 1: 'Malware'})

                st.success("✅ Prediction completed.")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name="inference_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ CSV must contain columns: " + ", ".join(feature_names))
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
