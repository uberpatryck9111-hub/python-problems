import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = {
    'Size': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    'bedrooms': [2, 3, 3, 4, 4, 5, 5, 6, 6],
    'age': [10, 5, 8, 3, 15, 2, 7, 1, 12],
    'price': [200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
}

df = pd. DataFrame(data)
print(df)

X = df[['Size', 'bedrooms', 'age']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"predictions: {y_pred}")
print(f"Actual prices: {y_test.values}")
print(f"Root Mean Squared Error: ${rmse:,.0f}")

plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

new_house = np.array([[2500, 4, 5]])
predicted_price = model.predict(new_house)
print(f"\nPredicted price for 2500sqft, 4 bed, 5 year old: ${predicted_price[0]:,.0f}")


