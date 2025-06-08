import streamlit as st
import yfinance as yf
import pandas as pd

st.title("BTC & Nasdaq 100 Swing Trading Agent")

tickers = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Nasdaq 100 (^NDX)": "^NDX"
}

selected = st.selectbox("Izaberi Asset:", list(tickers.keys()))
ticker = tickers[selected]

data = yf.download(ticker, period="90d", interval="1d")
data['SMA20'] = data['Close'].rolling(window=20).mean()

st.line_chart(data[['Close', 'SMA20']])
