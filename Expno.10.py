import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = {
    'make': ['Ford', 'Toyota', 'Ford', 'BMW', 'Toyota', 'BMW', 'Ford', 'Toyota', 'BMW', 'Toyota'],
    'model': ['Fiesta', 'Camry', 'Focus', '3 Series', 'Corolla', '5 Series', 'Mustang', 'Yaris', 'X5', 'Prius'],
    'year': [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
    'engine_size': [1.6, 2.0, 1.8, 3.0, 1.8, 2.5, 3.7, 1.5, 3.0, 1.8],
    'num_doors': [4, 4, 4, 4, 4, 4, 2, 4, 4, 4],
    'price': [7000, 15000, 12000, 25000, 16000, 27000, 30000, 14000, 35000, 22000]
}

df = pd.DataFrame(data)

print("First five rows of the dataset:")
print(df.head())

print("\nBasic statistical computations:")
print(df.describe())

print("\nColumns and their data types:")
print(df.dtypes)

if df.isnull().sum().any():
    for column in df.columns:
        if df[column].isnull().any():
            mode_value = df[column].mode()[0]
            df[column].fillna(mode_value, inplace=True)
print("\nNull values after replacement (if any):")
print(df.isnull().sum())

df_encoded = pd.get_dummies(df, drop_first=True)

plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm')
plt.show()

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model:", accuracy)
