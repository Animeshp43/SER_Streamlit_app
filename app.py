import streamlit as st
import numpy as np
import librosa
import os
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load model and encoder
model = load_model("cnn_lstm_ser_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Feature extraction
def extract_mfcc_sequence(path, max_len=173):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc.T

# UI
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload an audio file (.wav) and predict the emotion")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    mfcc = extract_mfcc_sequence("temp.wav")
    mfcc = np.expand_dims(mfcc, axis=0)

    prediction = model.predict(mfcc)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    st.success(f"🔮 Predicted Emotion: **{predicted_label.upper()}**")
