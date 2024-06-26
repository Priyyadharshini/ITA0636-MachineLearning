import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = pd.read_csv('C:/Users/priyy/Documents/Iris.csv')

plt.figure(figsize=(10, 6))
species_colors = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}
colors = iris['species'].map(species_colors)
plt.scatter(iris['sepal_width'], iris['sepal_length'], c=colors, s=50)
plt.title('Sepal Width vs Sepal Length')
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.show()

X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

new_data = [[5, 3, 1, 0.3]]
predicted_species = model.predict(new_data)

print("Predicted species:", predicted_species[0])
