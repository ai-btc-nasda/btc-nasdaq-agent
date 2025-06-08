
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

# Učitavanje podataka
try:
    data = yf.download(ticker, period="90d", interval="1d")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if not data.empty and 'Close' in data.columns:
        # SMA20
        data['SMA20'] = data['Close'].rolling(window=20).mean()

        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        sma = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        data['Upper Band'] = sma + (std * 2)
        data['Lower Band'] = sma - (std * 2)

        # Grafovi
        st.subheader("Cene + SMA20 + Bollinger Bands")
        st.line_chart(data[['Close', 'SMA20', 'Upper Band', 'Lower Band']])

        st.subheader("RSI")
        st.line_chart(data[['RSI']])

        st.subheader("MACD")
        st.line_chart(data[['MACD', 'Signal Line']])

    else:
        st.error("⚠️ Podaci nisu dostupni za izabrani asset. Pokušaj ponovo kasnije.")

except Exception as e:
    st.error(f"❌ Greška pri učitavanju podataka: {e}")
