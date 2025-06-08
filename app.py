import streamlit as st
import yfinance as yf
import pandas as pd

st.title("BTC & Nasdaq 100 Swing Trading Agent")

# Mapa tickera
tickers = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Nasdaq 100 (^NDX)": "^NDX"
}

# Odabir tickera
selected = st.selectbox("Izaberi Asset:", list(tickers.keys()))
ticker = tickers[selected]

# Učitavanje podataka
try:
    data = yf.download(ticker, period="90d", interval="1d")

    if not data.empty and 'Close' in data.columns:
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        st.line_chart(data[['Close', 'SMA20']])
    else:
        st.error("⚠️ Podaci nisu dostupni za izabrani asset. Pokušaj ponovo kasnije.")
except Exception as e:
    st.error(f"❌ Greška pri učitavanju podataka: {e}")
