import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# a) Read the Mobile price dataset using the Pandas module
df = pd.read_csv('C:/Users/priyy/Documents/mobile_price_dataset.csv')

# b) Print the 1st five rows
print(df.head())

# c) Basic statistical computations on the data set or distribution of data
print(df.describe())

# d) The columns and their data types
print(df.dtypes)

# e) Detect null values in the dataset. If there are any null values, replace them with mode value
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.mode().iloc[0])

# f) Explore the dataset using heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# g) Split the data into test and train
X = df.drop('price_range', axis=1)
y = df['price_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# h) Fit into the model Naive Bayes Classifier
model = GaussianNB()
model.fit(X_train, y_train)

# i) Predict the model
y_pred = model.predict(X_test)

# j) Find the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
