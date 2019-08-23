import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from matplotlib.dates import date2num
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

dates  = []
prices = []

def convert_dates(x):
    return datetime.strptime(x, "%d-%m-%Y").strftime("%Y-%m-%d")

df = pd.read_csv('HistoricalQuotes.csv')
df['date'] = pd.to_datetime(df['date'].apply(convert_dates)).apply(date2num)

for index, row in df.iterrows():
    dates.append(row[0])
    prices.append(float(row[1]))

dates = np.reshape(dates,(len(dates),1))

# Modelling
lin  = LinearRegression()
svr_rbf  = SVR(kernel='rbf', C=1e3, gamma=0.01)
X_train, X_test, y_train, y_test =  train_test_split(dates, prices, test_size=0.3, random_state=42)

lin.fit(X_train, y_train)
svr_rbf.fit(X_train, y_train)

pred_brf = svr_rbf.predict(X_test)
pred_lin = lin.predict(X_test)

lin.fit(dates, prices)
svr_rbf.fit(dates, prices)

fig, ax = plt.subplots()
ax.scatter(dates, prices, color='black', label='Data')
plt.plot(dates, lin.predict(dates), color='green', label='Linear Model')
plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF Model')
ax.xaxis_date()


plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=40)
plt.title('Apple Stock Exchange')
plt.legend()
plt.show()

new_date = '2019-06-27'
new_date1 = date2num(datetime.strptime(new_date, "%Y-%m-%d"))
ans_brf = svr_rbf.predict([[new_date1]])
print("RBF Prediction: "+str(round(ans_brf[0],2)))
ans_lin = lin.predict([[new_date1]])
print("Linear Prediction: "+str(round(ans_lin[0],2)))