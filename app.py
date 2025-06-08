
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("BTC & Nasdaq 100 Swing Trading Agent")

tickers = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Nasdaq 100 (^NDX)": "^NDX"
}

selected = st.selectbox("Izaberi Asset:", list(tickers.keys()))
ticker = tickers[selected]

try:
    data = yf.download(ticker, period="90d", interval="1d")

    # Ako multi-index, pojednostavi kolone
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if not data.empty and 'Close' in data.columns:
        # SMA20
        data['SMA20'] = data['Close'].rolling(window=20).mean()

        # RSI calculation
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # Buy/Sell/Hold signal na osnovu RSI
        data['Signal'] = np.where(data['RSI'] < 30, 'Buy',
                          np.where(data['RSI'] > 70, 'Sell', 'Hold'))

        # Prikaži grafik Close, SMA20
        st.line_chart(data[['Close', 'SMA20']])

        # Prikaži RSI
        st.line_chart(data['RSI'])

        # Prikaz signala za poslednji dan
        st.write(f"Signal za poslednji dan ({data.index[-1].date()}): **{data['Signal'].iloc[-1]}**")
    else:
        st.error("⚠️ Podaci nisu dostupni za izabrani asset. Pokušaj ponovo kasnije.")
except Exception as e:
    st.error(f"❌ Greška pri učitavanju podataka: {e}")
