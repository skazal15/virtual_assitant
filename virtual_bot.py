import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import date
#import openai
from Bard import Chatbot
import telegram
import time
import os
from telegram.ext import CommandHandler, MessageHandler, Filters, Updater

TOKEN_BOT = "1660877363:AAFd3WlLFfSMm0Hr0FR0aT6ZEpFFuF_Dbc0"
#TOKEN_OPENAPI = "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"
os.environ['BARD_API_KEY'] = "WgiAqPKTBRsMVbfqkHAaRmeXCfNxlUtBvCmmGl0RxfPFlUIK9BLym7gxsYEDzdFQyFL-aw."

bot = telegram.Bot(token=TOKEN_BOT)
#openai.api_key = TOKEN_OPENAPI

def analysis(SAHAM):
    saham = yf.download(SAHAM, period='max')
    saham = saham.dropna()
    saham_norm = (saham - np.min(saham)) / (np.max(saham) - np.min(saham))
    
    # Membuat model ARIMA dengan order (1, 1, 1)
    model = ARIMA(saham_norm['Close'], order=(1, 1, 1))
    results = model.fit()
    
    # Membuat prediksi untuk 30 hari ke depan
    forecast = results.predict(start=len(saham_norm), end=len(saham_norm)+29, dynamic=False)
    
    # Denormalisasi data
    forecast = forecast * (np.max(saham['Close']) - np.min(saham['Close'])) + np.min(saham['Close'])
    
    # Menghitung Mean Absolute Error (MAE)
    mae = np.mean(np.abs(forecast - saham['Close'].values[-30:]))
    print("MAE:", mae)
    
    # Gunakan hasil prediksi dan evaluasi model prediksi untuk membuat keputusan investasi
    if forecast.values[-1] > saham['Close'].values[-1]:
        return f"Beli saham {SAHAM}"
    else:
        return f"Jangan beli saham {SAHAM}"
    
# Membuat data latih dan target
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset)-time_steps-1):
        a = dataset[i:(i+time_steps), 0]
        X.append(a)
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)
    
def analysis_crypto(CRYPTO):
    crypto = yf.download(CRYPTO,period='max')
    # Menghitung Moving Average
    crypto['MA10'] = crypto['Close'].rolling(window=10).mean()
    crypto['MA20'] = crypto['Close'].rolling(window=20).mean()
    crypto['MA50'] = crypto['Close'].rolling(window=50).mean()

    # Membuat data training dan testing
    training_data = crypto.iloc[:-30, :]
    testing_data = crypto.iloc[-30:, :]
    
    # Normalisasi data
    scaler = MinMaxScaler()
    training_data = scaler.fit_transform(training_data[['Close']])
    testing_data = scaler.transform(testing_data[['Close']])


    time_steps = 30
    X_train, y_train = create_dataset(training_data, time_steps)
    X_test, y_test = create_dataset(testing_data, time_steps)

    # Membuat model LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Memprediksi harga crypto
    last_30_days = scaler.transform(crypto.tail(30)[['Close']])
    X_new = []
    X_new.append(last_30_days)
    X_new = np.array(X_new)
    X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))
    predicted_price = model.predict(X_new)
    predicted_price = scaler.inverse_transform(predicted_price)

    # Rekomendasi beli atau tidak
    current_price = crypto['Close'].iloc[-1]
    if predicted_price > current_price:
         return f'Rekomendasi: BELI {CRYPTO}'
    else:
        return f'Rekomendasi: JANGAN BELI {CRYPTO}'



def analysis_random(SAHAM):
    saham = yf.download(SAHAM,start="2015-01-01", end=date.today())
    # Membuat fitur-fitur yang akan digunakan untuk prediksi
    saham['PriceDiff'] = saham['Close'].shift(-1) - saham['Close']
    saham['Open-Close'] = saham['Open'] - saham['Close'].shift(-1)
    saham['High-Low'] = saham['High'] - saham['Low']
    saham['PercentChange'] = saham['Close'].pct_change()
    saham = saham.dropna()

    # Memisahkan fitur dan label
    X = saham[['Open', 'High', 'Low', 'Volume', 'PriceDiff', 'Open-Close', 'High-Low', 'PercentChange']]
    y = np.where(saham['PriceDiff'].shift(-1) > 0, 1, 0)

    # Membagi data menjadi data training dan data testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Melatih model dengan algoritma Random Forest Regression
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Memprediksi nilai saham di masa depan
    latest_data = saham.tail(1)[['Open', 'High', 'Low', 'Volume', 'PriceDiff', 'Open-Close', 'High-Low', 'PercentChange']]
    predictions = rf.predict(latest_data)

    # Gunakan hasil prediksi untuk membuat keputusan investasi
    if predictions[0] > 0.5:
        return f"Beli saham {SAHAM}"
    else:
        return f"Jangan beli saham {SAHAM}"



# Fungsi untuk menangani perintah '/start'
def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Halo! Saya adalah Shafar.")

# Fungsi untuk menangani pesan yang diterima oleh bot
def reply_to_message(update, context):
    message_text = update.message.text
    response = generate_response(message_text)
    context.bot.send_message(chat_id=update.effective_chat.id, text=response)
    time.sleep(1) # memberi jeda waktu 1 detik

# Fungsi untuk menghasilkan respon dari OpenAI Large Language Model
def generate_response(input_text):
    if ".JK" in input_text:
        ticker = input_text.split(" ")

        if "arima" in input_text:
            arima = analysis(ticker[1])
            return arima
        
        if "rforest" in input_text:
            rforest = analysis_random(ticker[1])
            return rforest
        
    if "-USD" in input_text:
        Crypto = analysis_crypto(input_text)
        return Crypto
        
    else:
        bard = os.environ.get('BARD_API_KEY')
        chatbot = Chatbot(bard)
        answer = chatbot.ask(input_text)['content']
        return answer


# Fungsi utama untuk menjalankan bot
def main():
    updater = Updater(token=TOKEN_BOT, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.text, reply_to_message))
    updater.start_polling()
    updater.idle()
    
if __name__ == '__main__':
    main()
