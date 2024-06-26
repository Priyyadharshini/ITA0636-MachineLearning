import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

data = {'text': ['I love this movie', 'This movie is terrible', 'I enjoyed this film', 'I hate this movie', 'Fantastic film', 'Worst film ever'],
        'label': ['positive', 'negative', 'positive', 'negative', 'positive', 'negative']}
df = pd.DataFrame(data)

X = df['text']
y = df['label']
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")
