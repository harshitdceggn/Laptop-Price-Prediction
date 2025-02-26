import streamlit as st
import pickle
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

st.title("💻 Laptop Price Predictor")

repo_id = "your-username/laptop-price-predictor"  

@st.cache_data
def load_files():
    model_path = hf_hub_download(repo_id=repo_id, filename="pipe.pkl")
    df_path = hf_hub_download(repo_id=repo_id, filename="df.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(df_path, "rb") as f:
        df = pickle.load(f)

    return model, df

pipe, df = load_files()

st.subheader("🔧 Select Laptop Specifications")

Brand = st.selectbox("Brand", df['Brand'].unique())
Processor = st.selectbox("Processor", df['Processor'].unique())
RAM = st.selectbox("RAM (GB)", df['RAM (GB)'].unique())
Storage = st.selectbox("Storage", df['Storage'].unique())
GPU = st.selectbox("GPU", df['GPU'].unique())
ScreenSize = st.selectbox("Screen Size (inch)", df['Screen Size (inch)'].unique())
Resolution = st.selectbox("Resolution", df['Resolution'].unique())

BatteryLife = st.number_input('Battery Life (Hours)', min_value=1.0, max_value=24.0, value=6.0)
Weight = st.number_input('Weight of the Laptop (Kg)', min_value=0.5, max_value=5.0, value=1.5)
OperatingSystem = st.selectbox("Operating System", df['Operating System'].unique())

if st.button('🔮 Predict Price'):
    try:
        query = np.array([Brand, Processor, RAM, Storage, GPU, ScreenSize, Resolution, BatteryLife, Weight, OperatingSystem])
        query = query.reshape(1, -1)

        predicted_price = np.exp(pipe.predict(query)[0])

        st.success(f"💰 Estimated Laptop Price: **₹{int(predicted_price):,}**")
    except Exception as e:
        st.error(f"⚠️ Prediction failed! Error: {e}")
