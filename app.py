
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
        # SMA 20
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
        data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Signal'] = np.where(data['MACD'] > data['Signal_Line'], 'Buy', 'Sell')

        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['BB_Std']
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['BB_Std']

        # Prikaz grafikona
        st.subheader("Cene i SMA20")
        st.line_chart(data[['Close', 'SMA20']])

        st.subheader("RSI")
        st.line_chart(data['RSI'])

        st.subheader("MACD i Signal linija")
        st.line_chart(data[['MACD', 'Signal_Line']])

        st.subheader("Bollinger Bands")
        st.line_chart(data[['Close', 'BB_Upper', 'BB_Lower']])

        # Alert na osnovu MACD signala
        last_macd_signal = data['MACD_Signal'].iloc[-1]
        if last_macd_signal == 'Buy':
            st.success("🚀 MACD signal za kupovinu!")
        elif last_macd_signal == 'Sell':
            st.warning("⚠️ MACD signal za prodaju!")

    else:
        st.error("⚠️ Podaci nisu dostupni za izabrani asset. Pokušaj ponovo kasnije.")

except Exception as e:
    st.error(f"❌ Greška pri učitavanju podataka: {e}")
