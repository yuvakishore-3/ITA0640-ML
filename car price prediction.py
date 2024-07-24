import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
data = pd.DataFrame({
    'Mileage': [10000, 20000, 30000, 40000, 50000],
    'Age': [1, 2, 3, 4, 5],
    'Price': [20000, 18000, 16000, 14000, 12000]  
})
X = data.drop('Price', axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
joblib.dump(model, 'car_price_predictor.pkl')
def generate_pdf(filename, mse, r2):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, 'Car Price Prediction Report')
    c.drawString(100, 730, f'Mean Squared Error: {mse}')
    c.drawString(100, 710, f'R^2 Score: {r2}')
    c.save()

generate_pdf('car_price_prediction_report.pdf', mse, r2)
