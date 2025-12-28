# Sentiment-analysis
Sentiment analysis of feedback comments using Natural Language Processing (NLP) and Naive Bayes classifier in Python.
Sentiment Analysis

A simple machine-learning project that performs sentiment classification on text feedback using:

Text cleaning (regex + stopwords + stemming)

Bag-of-Words (CountVectorizer)

Naive Bayes classifier

Model evaluation (accuracy, confusion matrix, classification report)

This project is ideal for beginners learning NLP + Machine Learning workflow.

📂 Project Structure
sentiment-analysis/
│
├── gr_feedback_dataset.csv
├── sentimentanalysis.py   (or your script name)
└── README.md



📦 Requirements

Install the libraries first:

pip install numpy pandas matplotlib scikit-learn nltk


And download NLTK stopwords (runs automatically in script, but you can do manually):

import nltk
nltk.download("stopwords")

🧠 How It Works
1️⃣ Load Dataset

Reads the CSV file:

dataset = pd.read_csv('gr_feedback_dataset.csv')


Dataset must contain:

column	description
comment	text / feedback
label	sentiment label (positive / negative / etc.)
2️⃣ Text Pre-Processing

Remove symbols/numbers

Convert to lowercase

Remove stopwords

Apply stemming

Creates a cleaned corpus for model training.

3️⃣ Feature Extraction (Bag of Words)
CountVectorizer(max_features=1500)


Converts text into numerical vectors.

4️⃣ Train/Test Split

20% data for testing.

5️⃣ Model — Naive Bayes
MultinomialNB()


A simple and fast classifier for text data.

6️⃣ Evaluation

The script prints:

✔ Classification Report
✔ Confusion Matrix
✔ Accuracy Score

So you can easily analyze performance.

▶️ Run the Project

Make sure you are inside the project folder:

python main.py

📊 Expected Output (Example)

Accuracy score

Precision/Recall/F1

Confusion matrix

🚀 Future Improvements

You can extend this project by adding:

TF-IDF features

Logistic Regression / SVM / Random Forests

Deep Learning (LSTM / BERT)

Streamlit or Flask web app interface

Model saving using pickle
