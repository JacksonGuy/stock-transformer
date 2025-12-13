import numpy as np
import pandas as pd
import yfinance as yf

from src.globals import INPUT_LENGTH, OUTPUT_LENGTH

tickers = [
    "AAPL", "AMZN", "META", "MSFT", "GOOG", "NVDA",
]

def get_data(tick, length="730d", interval="1d"):
    data = yf.download(tick, period=length, interval=interval, auto_adjust=True, progress=False)

    close = np.array(data["Close"]).flatten()
    open = np.array(data["Open"]).flatten()
    high = np.array(data["High"]).flatten()
    low = np.array(data["Low"]).flatten()

    ticker_df = pd.DataFrame({
        tick+"_open": open,
        tick+"_high": high,
        tick+"_low": low,
        tick+"_close": close,
    }, index=data.index)

    return ticker_df

def get_full_history(length="730d", interval="1h"):
    data_frames = []

    for symbol in tickers:
        df = get_data(symbol, length, interval)
        data_frames.append(df)

    # Combine all ticker frames
    df = pd.concat(data_frames, axis=1)

    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove rows with NaN
    df.dropna(inplace=True)

    return df

def sequence_dataframe(df):
    sequences, labels = [], []

    for i in range(len(df) - (INPUT_LENGTH + OUTPUT_LENGTH)):
        sequences.append(df.iloc[i:i+INPUT_LENGTH].values)
        labels.append(df.iloc[i+INPUT_LENGTH:i+INPUT_LENGTH+OUTPUT_LENGTH].values)

    return np.array(sequences), np.array(labels)

def sequence_dataframe_relative(df):
    sequences, labels = [], []

    for i in range(len(df) - (INPUT_LENGTH + OUTPUT_LENGTH + 1)):
        sequences.append(df.iloc[i:i+INPUT_LENGTH+1].values)
        labels.append(df.iloc[i+INPUT_LENGTH:i+INPUT_LENGTH+OUTPUT_LENGTH+1].values)

    return np.array(sequences), np.array(labels)

def relative_change(sequences):
    result = sequences.copy()

    for i in range(sequences.shape[0]):
        for j in range(1, sequences.shape[1]):
            result[i, j, :] = (sequences[i, j, :] - sequences[i, j-1, :]) / sequences[i, j-1, :]

    return result[:, 1:, :]

def normalize_per_window_per_stock(x):
    x_stocks = x.reshape(x.shape[0], x.shape[1], len(tickers), 4)

    mean = x_stocks.mean(axis=1, keepdims=True)
    stdev = x_stocks.std(axis=1, keepdims=True) + 1e-6

    x_norm = (x_stocks - mean) / stdev

    x_norm = x_norm.reshape(x.shape)

    return x_norm, mean, stdev

def denormalize(x_norm, mean, stdev):
    x_norm = x_norm.reshape(x_norm.shape[0], x_norm.shape[1], len(tickers), 4)

    x_raw = x_norm * stdev + mean

    x_raw = x_raw.reshape(x_norm.shape)

    return x_raw