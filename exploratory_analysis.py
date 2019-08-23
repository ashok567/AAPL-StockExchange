import pandas as pd
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from matplotlib.dates import date2num
from datetime import datetime

def convert_dates(x):
    return datetime.strptime(x, "%d-%m-%Y").strftime("%Y-%m-%d")

def settings():
    plt.plot(df['date'], df['sma10'], c='blue', label='SMA over 10 period')
    plt.plot(df['date'], df['sma50'], c='black', label='SMA over 50 period')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Apple Stock Exchange')
    plt.legend()
    plt.tight_layout()

df = pd.read_csv('HistoricalQuotes.csv')
df['date'] = pd.to_datetime(df['date'].apply(convert_dates))

df['sma10'] = df['close'].rolling(10).mean()
df['sma50'] = df['close'].rolling(50).mean()

ohlc = df[['date', 'open', 'high', 'low', 'close']].copy()
ohlc['date'] = ohlc['date'].apply(date2num)

fig, ax = plt.subplots()

candlestick_ohlc(ax, ohlc.values, width=0.9, colorup='g', colordown='r')

ax.xaxis_date()
settings()
plt.savefig('ohlc.png')
plt.show()


plt.plot(df['date'], df['close'], c='green', label='Closing price')
settings()
plt.savefig('time series.png')
plt.show()