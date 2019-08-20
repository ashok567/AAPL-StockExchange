import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('HistoricalQuotes-1Month.csv')

fig, ax = plt.subplots()


def candlestick(ax, df, width=0.2, colorup='k', colordown='r', linewidth=0.5):

    date = np.array(df['date'])
    color = []

    for i in range(1,len(date)): 
        if df['open'][i-1]<df['close'][i]:
            color.append('k')
        else:
            color.append('r')
    
    # offset = .4

    #high low Lines
    ax.vlines(date,df['low'],df['high'],color)

    #open Lines
    ax.hlines(df['open'],date,date,color)
    
    #close Lines
    ax.hlines(df['close'],date,date,color)

    ax.autoscale_view()

    return

candlestick(ax, df)
plt.xticks(rotation=70)
plt.show()