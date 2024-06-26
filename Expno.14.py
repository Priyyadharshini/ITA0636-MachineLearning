import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('C:/Users/priyy/Documents/HousePricePrediction.csv')
print("First five rows of the dataset:")
print(df.head())

print("\nBasic statistical summary:")
print(df.describe())

print("\nData types of the columns:")
print(df.dtypes)

print("\nDetecting null values:")
print(df.isnull().sum())

for column in df.columns:
    if df[column].isnull().sum() > 0:
        df[column].fillna(df[column].mode()[0], inplace=True)

print("\nNull values after cleaning:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

X = df.drop('price', axis=1)
y = df['price']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

print("\nPredicted prices for test set:")
print(y_pred)
