import streamlit as st
import pickle
import numpy as np

#import the model
pipe=pickle.load(open('pipe.pkl','rb'))
df= pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

Brand= st.selectbox("Brand",df['Brand'].unique())
Processor=st.selectbox("Processor",df['Processor'].unique())
RAM=st.selectbox("RAM (GB)",df['RAM (GB)'].unique())
Storage=st.selectbox("Storage",df['Storage'].unique())
GPU=st.selectbox("GPU",df['GPU'].unique())
ScreenSize=st.selectbox("Screen Size (inch)",df['Screen Size (inch)'].unique())
Resolution=st.selectbox("Resolution",df['Resolution'].unique())
BatteryLife=st.number_input('battery of the laptop')
Weight=st.number_input('Weight of the laptop')
OperatingSystem=st.selectbox("Operating System",df['Operating System'].unique())
if st.button('Predict price'):
    query=np.array([Brand,Processor,RAM,Storage,GPU,ScreenSize,Resolution,BatteryLife,Weight,OperatingSystem])
    query=query.reshape(1,10)
    st.title(int(np.exp(pipe.predict(query)[0])))
    