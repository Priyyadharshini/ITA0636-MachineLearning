import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = {
    'month': pd.date_range(start='1/1/2019', periods=60, freq='M'),
    'advertising': [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8,
                    66.1, 214.7, 23.8, 97.5, 204.1, 195.4, 67.8, 281.4, 69.2, 147.3,
                    230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8,
                    66.1, 214.7, 23.8, 97.5, 204.1, 195.4, 67.8, 281.4, 69.2, 147.3,
                    230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8,
                    66.1, 214.7, 23.8, 97.5, 204.1, 195.4, 67.8, 281.4, 69.2, 147.3],
    'sales': [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 14.6, 24.4,
              23.8, 14.1, 15.6, 12.6, 12.2, 11.7, 15.5, 15.9, 19.6, 14.8,
              22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 14.6, 24.4,
              23.8, 14.1, 15.6, 12.6, 12.2, 11.7, 15.5, 15.9, 19.6, 14.8,
              22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 14.6, 24.4,
              23.8, 14.1, 15.6, 12.6, 12.2, 11.7, 15.5, 15.9, 19.6, 14.8]
}
data = pd.DataFrame(data)

print("First five rows of the dataset:")
print(data.head())

print("\nBasic statistical summary:")
print(data.describe())

print("\nData types of the columns:")
print(data.dtypes)

plt.scatter(data['advertising'], data['sales'])
plt.xlabel('Advertising Expenditure')
plt.ylabel('Sales')
plt.title('Relationship between Advertising Expenditure and Sales')
plt.show()

print("\nDetecting null values:")
print(data.isnull().sum())
data = data.fillna(data.mode().iloc[0])
print("\nNull values after cleaning:")
print(data.isnull().sum())

X = data[['advertising']]  
y = data['sales']         
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

print("\nPredicted sales for the test set:")
print(y_pred)
