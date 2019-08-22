import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from mpl_finance import candlestick_ohlc
from matplotlib.dates import date2num
from datetime import datetime


def convert_dates(x):
    return datetime.strptime(x, "%d-%m-%Y").strftime("%Y-%m-%d")

df = pd.read_csv('HistoricalQuotes-3Months.csv')
df['date'] = pd.to_datetime(df['date'].apply(convert_dates))

ohlc = df[['date', 'open', 'high', 'low', 'close']].copy()
ohlc['date'] = ohlc['date'].apply(date2num)

fig, ax = plt.subplots()
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='g', colordown='r')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Apple Stock Exchange')
ax.xaxis_date()
plt.xticks(rotation=40)
plt.tight_layout()
plt.show()
plt.savefig('ohlc.png')