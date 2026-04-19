
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACDIndicator

model  = joblib.load('crypto_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Crypto price movement predictor")
st.write("Predicts whether tomorrow's price will go UP or DOWN.")

ticker = st.selectbox("Choose a coin", ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"])

if st.button("Predict"):
    with st.spinner("Fetching data..."):
        df = yf.download(ticker, period="3mo", auto_adjust=True)
        df.columns = df.columns.get_level_values(0)
        close = df["Close"].squeeze()

        df["RSI"]   = RSIIndicator(close=close).rsi()
        macd        = MACDIndicator(close=close)
        df["MACD"]  = macd.macd()
        df["MA_10"] = close.rolling(10).mean()
        df["MA_50"] = close.rolling(50).mean()
        df.dropna(inplace=True)

        features = ["Close", "RSI", "MACD", "MA_10", "MA_50"]
        latest = df[features].iloc[[-1]]
        scaled = scaler.transform(latest)
        pred   = model.predict(scaled)[0]
        prob   = model.predict_proba(scaled)[0]

        st.metric("Current price", f"${close.iloc[-1]:,.2f}")

        if pred == 1:
            st.success(f"Prediction: price will go UP tomorrow ({round(prob[1]*100)}% confidence)")
        else:
            st.error(f"Prediction: price will go DOWN tomorrow ({round(prob[0]*100)}% confidence)")
