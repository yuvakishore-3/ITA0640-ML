import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
sales = np.random.poisson(lam=200, size=100) + np.linspace(0, 10, 100)
data = pd.DataFrame({'Date': dates, 'Sales': sales})
data['Day'] = data.index
X = data[['Day']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
future_days = pd.DataFrame({'Day': range(len(data), len(data) + 10)})
future_sales = model.predict(future_days)
print("Future sales predictions:", future_sales)
