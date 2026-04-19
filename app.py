
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import time
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACDIndicator
from newsapi import NewsApiClient
from textblob import TextBlob
from dotenv import load_dotenv

BASE_DIR = "/Users/myapple/crypto-ml-project"
load_dotenv(os.path.join(BASE_DIR, ".env"))
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

model  = joblib.load(os.path.join(BASE_DIR, "crypto_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

COIN_NAMES = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "BNB-USD": "Binance coin",
    "SOL-USD": "Solana",
    "XRP-USD": "XRP",
    "ADA-USD": "Cardano",
    "DOGE-USD": "Dogecoin",
    "AVAX-USD": "Avalanche"
}

UP_SOUND = """
<script>
function playUpSound() {
    const ctx = new AudioContext();
    const frequencies = [523, 659, 784];
    frequencies.forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = freq;
        osc.type = "sine";
        gain.gain.setValueAtTime(0.3, ctx.currentTime + i * 0.15);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.15 + 0.3);
        osc.start(ctx.currentTime + i * 0.15);
        osc.stop(ctx.currentTime + i * 0.15 + 0.3);
    });
}
playUpSound();
</script>
"""

DOWN_SOUND = """
<script>
function playDownSound() {
    const ctx = new AudioContext();
    const frequencies = [784, 659, 523];
    frequencies.forEach((freq, i) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.frequency.value = freq;
        osc.type = "sine";
        gain.gain.setValueAtTime(0.3, ctx.currentTime + i * 0.15);
        gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.15 + 0.3);
        osc.start(ctx.currentTime + i * 0.15);
        osc.stop(ctx.currentTime + i * 0.15 + 0.3);
    });
}
playDownSound();
</script>
"""

def get_sentiment(coin_name):
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(
            q=coin_name,
            language="en",
            sort_by="publishedAt",
            page_size=10
        )
        headlines = [a["title"] for a in articles["articles"]]
        if not headlines:
            return 0, "Neutral", []
        scores = [TextBlob(h).sentiment.polarity for h in headlines]
        avg = sum(scores) / len(scores)
        mood = "Positive" if avg > 0.1 else "Negative" if avg < -0.1 else "Neutral"
        return round(avg, 3), mood, headlines[:3]
    except:
        return 0, "Unavailable", []

def run_prediction(ticker):
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
    confidence = max(prob)

    coin_name = COIN_NAMES[ticker]
    sentiment_score, mood, headlines = get_sentiment(coin_name)

    return df, close, pred, prob, confidence, sentiment_score, mood, headlines

st.title("Crypto price movement predictor")
st.write("Auto-refreshes every 5 minutes. Plays sound when confidence is above 80%.")

ticker = st.selectbox("Choose a coin", list(COIN_NAMES.keys()))

REFRESH_INTERVAL = 300

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None

time_since = time.time() - st.session_state.last_refresh
should_refresh = time_since >= REFRESH_INTERVAL

col_btn, col_timer = st.columns([1, 3])
with col_btn:
    manual = st.button("Predict now")
with col_timer:
    if st.session_state.last_refresh > 0:
        remaining = max(0, int(REFRESH_INTERVAL - time_since))
        st.caption(f"Next auto-refresh in {remaining}s")

if manual or should_refresh:
    with st.spinner("Fetching live data and news..."):
        result = run_prediction(ticker)
        st.session_state.pred_result = result
        st.session_state.last_refresh = time.time()

if st.session_state.pred_result:
    df, close, pred, prob, confidence, sentiment_score, mood, headlines = st.session_state.pred_result

    st.metric("Current price", f"${close.iloc[-1]:,.2f}")
    st.write("---")

    if confidence < 0.65:
        st.warning(f"Model confidence is only {round(confidence*100)}% — not confident enough to predict today.")
    elif pred == 1:
        st.success(f"Prediction: price will go UP tomorrow ({round(prob[1]*100)}% confidence)")
        if confidence >= 0.80:
            st.info("Strong signal detected — confidence above 80%!")
            st.components.v1.html(UP_SOUND, height=0)
    else:
        st.error(f"Prediction: price will go DOWN tomorrow ({round(prob[0]*100)}% confidence)")
        if confidence >= 0.80:
            st.info("Strong signal detected — confidence above 80%!")
            st.components.v1.html(DOWN_SOUND, height=0)

    st.write("---")
    st.subheader("News sentiment")
    col1, col2 = st.columns(2)
    col1.metric("Sentiment score", sentiment_score)
    col2.metric("Market mood", mood)

    if headlines:
        st.write("Latest headlines:")
        for h in headlines:
            st.write(f"- {h}")

    st.write("---")
    st.subheader("Recent price trend")
    st.line_chart(df["Close"].tail(30))

    st.subheader("Last 5 days indicators")
    st.dataframe(df[["Close", "RSI", "MACD", "MA_10", "MA_50"]].tail())

time.sleep(10)
st.rerun()
