import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib

# Define shared constants
feature_names = ['spyware', 'encrypter', 'downloader', 'backdoor', 'ransomware']

# Model definition
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

# Load model and scaler dynamically
@st.cache_resource
def load_model_and_scaler(model_type):
    if model_type == 'Obfuscated':
        model_path = 'models/mlp_model_obfuscated.pt'
        scaler_path = 'models/scaler_obfuscated.pkl'
    else:
        model_path = 'models/mlp_model_final.pt'
        scaler_path = 'models/scaler.pkl'

    model = MLP(input_dim=5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

# UI starts here
st.title("🧠 Malware Detection using MLP (5 Features)")

model_type = st.radio("Choose model version:", ['Original', 'Obfuscated'])
model, scaler = load_model_and_scaler(model_type)

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
    uploaded_file = st.file_uploader("Upload X_inference.csv (features)", type="csv")
    uploaded_labels = st.file_uploader("Upload y_inference.csv (true labels)", type="csv")

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

                if uploaded_labels is not None:
                    y_true = pd.read_csv(uploaded_labels).values.flatten()
                    if len(y_true) == len(predictions):
                        correct = (y_true == predictions).sum()
                        total = len(y_true)
                        accuracy = correct / total * 100
                        st.success(f"✅ Prediction completed: {correct}/{total} correct ({accuracy:.2f}%)")
                    else:
                        st.warning("⚠️ Mismatch in number of rows between X and y inference files.")
                else:
                    st.success("✅ Prediction completed.")

                st.dataframe(df.head(10))

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
