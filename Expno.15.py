import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

df = pd.read_csv("C:/Users/priyy/Documents/breastcancer.csv")

print("First five rows of the dataset:")
print(df.head())

print("\nBasic statistical computations:")
print(df.describe())

print("\nColumns and their data types:")
print(df.dtypes)

print("\nDetecting null values:")
print(df.isnull().sum())

for column in df.columns:
    mode_value = df[column].mode()[0]
    df[column].fillna(mode_value, inplace=True)

print("\nNull values after handling:")
print(df.isnull().sum())

X = df.drop('Class', axis=1)  
y = df['Class'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")

print(classification_report(y_test, y_pred))
