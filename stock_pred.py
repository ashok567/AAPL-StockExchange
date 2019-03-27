import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression 

dates  = []
prices = []

df = pd.read_csv('HistoricalQuotes.csv')

# for index, row in df.iterrows():
#     dates.append(int(row[0].replace('-','')))
#     prices.append(float(row[1]))

# dates = np.reshape(dates,(len(dates),1))

dates = df[['date', 'open']]
prices = df['close']

# Modelling
lin  = LinearRegression()
svr_rbf  = SVR(kernel='rbf', C=1e3, gamma=0.1)

lin.fit(dates, prices)
svr_rbf.fit(dates, prices)


plt.scatter(dates, prices, color='black', label='Data')
plt.plot(dates, lin.predict(dates), color='green', label='Linear Model')
plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')


plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Apple Stock Exchange')
plt.legend()
plt.show()


ans_brf = svr_rbf.predict(27032019)
print(ans_brf)
ans_lin = lin.predict(27032019)
print(ans_lin)