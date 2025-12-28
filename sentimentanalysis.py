
# ==============================
#  Import Libraries
# ==============================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==============================
#  Read Dataset
# ==============================
dataset = pd.read_csv('gr_feedback_dataset.csv')
print("\n--- Dataset Head ---")
print(dataset.head())

print("\n--- Missing Values ---")
print(dataset.isnull().sum())


# ==============================
#  Text Cleaning
# ==============================
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['comment'][i])  # Remove symbols
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# ==============================
#  Bag of Words Model
# ==============================
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

y = dataset['label'].values

# ==============================
#  Train-Test Split
# ==============================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.20, random_state = 0
)

# ==============================
#  ONLY Naive Bayes Model
# ==============================
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# ==============================
#  Predictions
# ==============================
y_pred = classifier.predict(X_test)

# ==============================
#  Evaluation
# ==============================
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))
