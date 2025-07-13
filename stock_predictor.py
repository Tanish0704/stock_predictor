# 📊 TCS Stock Price Predictor - Linear Regression Model
# ✨ Created by: Tanish Tomar | Codec AI Internship

# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime as dt

# Step 1: Automatically get today's date
today = dt.date.today()
print(f"📅 Today's date: {today}")

# Step 2: Download TCS stock data from Jan 1, 2022 to today
data = yf.download('TCS.NS', start='2022-01-01', end=today.isoformat())
data.reset_index(inplace=True)

# Step 3: Prepare the dataset
df = data[['Date', 'Close']].copy()
df['Date'] = pd.to_datetime(df['Date'])
df['DateAsNumber'] = df['Date'].apply(lambda x: x.toordinal())

# Step 4: Define features and target variable
X = df[['DateAsNumber']]
y = df['Close']

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101
)

# Step 6: Train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 7: Make predictions and evaluate the model
predictions = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"📉 RMSE for TCS: ₹{rmse:.2f}")

# Step 8: Visualize actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual Prices', alpha=0.6)
plt.plot(X_test, predictions, color='red', label='Predicted Prices')
plt.xlabel('Date (ordinal)')
plt.ylabel('TCS Closing Price (₹)')
plt.title('TCS Stock Price Prediction up to July 2025')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 9: Predict price for 14th July 2025
predict_date = dt.date(2025, 7, 14).toordinal()
predicted_price = regressor.predict([[predict_date]])
print(f"📅 Predicted TCS Price for 14-07-2025: ₹{predicted_price[0][0]:.2f}")