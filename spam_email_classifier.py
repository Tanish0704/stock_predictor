# 📩 Spam Email Classifier
# ✨ Created by: Tanish Tomar | Codec AI Internship


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import requests


url = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/spam.csv"
try:
    df = pd.read_csv("spam.csv", encoding='latin-1')
except FileNotFoundError:
    print("Downloading spam.csv...")
    response = requests.get(url)
    with open("spam.csv", "wb") as f:
        f.write(response.content)
    df = pd.read_csv("spam.csv", encoding='latin-1')


df = df[['label', 'text']]  # Keep only necessary columns
df.columns = ['label', 'message']  # Rename columns

df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})


X = df['message']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

print(f"✅ Accuracy: {accuracy*100:.2f}%")
print("🧾 Confusion Matrix:")
print(conf_mat)


sample = ["Congratulations! You've won a free iPhone. Click now!"]
sample_vec = vectorizer.transform(sample)
prediction = model.predict(sample_vec)

print("\n📬 Sample Message:", sample[0])
print("📌 Prediction:", "Spam" if prediction[0] else "Not Spam")