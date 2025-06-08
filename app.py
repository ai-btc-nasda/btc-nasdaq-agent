
    
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

st.title("BTC & Nasdaq 100 Swing Trading Agent")

tickers = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Nasdaq 100 (^NDX)": "^NDX"
}

selected = st.selectbox("Izaberi Asset:", list(tickers.keys()))
ticker = tickers[selected]

# Učitavanje podataka
try:
    data = yf.download(ticker, period="180d", interval="1d")

    # Pojednostavi kolone ako multiindex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if not data.empty and 'Close' in data.columns:
        # --- TEHNIČKI INDIKATORI ---
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
        ema12 = data['Close'].ewm(span=12, adjust=False).mean()
        ema26 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema12 - ema26
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        data['BB_MA20'] = data['Close'].rolling(window=20).mean()
        data['BB_std'] = data['Close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_MA20'] + 2 * data['BB_std']
        data['BB_lower'] = data['BB_MA20'] - 2 * data['BB_std']

        st.subheader("Tehnički indikatori")
        st.line_chart(data[['Close', 'SMA20']])
        st.line_chart(data[['RSI']])
        st.line_chart(data[['MACD', 'MACD_signal']])
        st.line_chart(data[['BB_upper', 'Close', 'BB_lower']])

    else:
        st.error("⚠️ Podaci nisu dostupni za izabrani asset. Pokušaj ponovo kasnije.")
except Exception as e:
    st.error(f"❌ Greška pri učitavanju podataka: {e}")

# --- LSTM MODEL ---

def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(data_close):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_close.values.reshape(-1, 1))

    look_back = 60
    X, y = create_dataset(data_scaled, look_back)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = build_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    return model, scaler, look_back

def predict_next_price(model, scaler, data_close, look_back):
    last_60 = data_close[-look_back:].values.reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60)
    input_data = last_60_scaled.reshape(1, look_back, 1)
    pred_scaled = model.predict(input_data)
    pred_price = scaler.inverse_transform(pred_scaled)
    return pred_price[0][0]

if st.button("Treniraj LSTM i predvidi sledeću cenu"):
    if not data.empty and 'Close' in data.columns:
        with st.spinner("Trening LSTM modela..."):
            model, scaler, look_back = train_lstm_model(data['Close'])
            prediction = predict_next_price(model, scaler, data['Close'], look_back)
            st.success(f"Predviđena cena za sledeći dan: {prediction:.2f} USD")
    else:
        st.error("Nema dovoljno podataka za treniranje LSTM modela.")
