import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split

dates  = []
prices = []

df = pd.read_csv('HistoricalQuotes-1Month.csv')

for index, row in df.iterrows():
    dates.append(int(row[0].replace('-','')))
    prices.append(float(row[1]))

dates = np.reshape(dates,(len(dates),1))

# Modelling
lin  = LinearRegression()
svr_rbf  = SVR(kernel='rbf', C=1e3, gamma=0.1)

X_train, X_test, y_train, y_test =  train_test_split(dates, prices, test_size=0.3, random_state=42)

lin.fit(X_train, y_train)
svr_rbf.fit(X_train, y_train)

pred_brf = svr_rbf.predict(X_test)
pred_lin = lin.predict(X_test)


# plt.scatter(y_test,[round(i,2) for i in pred_brf])
# plt.xlabel('Y Test')
# plt.ylabel('Predicted Y')
# plt.title('RBF Prediction')
# plt.show()

# plt.scatter(y_test,[round(i,2) for i in pred_lin])
# plt.xlabel('Y Test')
# plt.ylabel('Predicted Y')
# plt.title('Linear Prediction')
# plt.show()

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


ans_brf = svr_rbf.predict(27)
print(round(ans_brf[0],2))
ans_lin = lin.predict(27)
print(round(ans_lin[0],2))