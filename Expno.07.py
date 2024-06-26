import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = {
    'Occupation': ['Engineer', 'Doctor', 'Artist', 'Engineer', 'Artist', 'Doctor', 'Engineer', 'Artist', 'Doctor', 'Engineer'],
    'Credit Score': [720, 680, 650, 700, 710, 690, 730, 640, 660, 750]
}
df = pd.DataFrame(data)
print("First five rows of the dataset:")
print(df.head())

print("\nBasic statistical computations:")
print(df.describe())

print("\nColumns and their data types:")
print(df.dtypes)

print("\nNull values in the dataset:")
print(df.isnull().sum())

df.at[2, 'Credit Score'] = None
print("\nNull values after insertion:")
print(df.isnull().sum())

mode_value = df['Credit Score'].mode()[0]
df['Credit Score'].fillna(mode_value, inplace=True)
print("\nDataset after handling null values:")
print(df)

sns.boxplot(x='Occupation', y='Credit Score', data=df)
plt.title('Credit Scores Based on Occupation')
plt.show()

X = pd.get_dummies(df['Occupation'], drop_first=True) 
y = df['Credit Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nPredicted values for the test set:")
print(y_pred)
print(f"Model Accuracy: {accuracy}")

