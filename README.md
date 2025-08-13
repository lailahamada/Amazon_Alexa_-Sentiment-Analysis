# Amazon Alexa Reviews Sentiment Analysis

## Project Overview
This project analyzes Amazon Alexa product reviews to predict customer sentiment (positive or negative). The dataset contains verified reviews along with feedback ratings. The goal is to preprocess the text data, convert it to numerical features, and train machine learning models for sentiment classification.

---

## Dataset
- **Source:** Amazon Alexa Reviews Dataset  
- **Features:**  
  - `verified_reviews`: The text of the review  
  - `feedback`: Sentiment label (0 = Negative, 1 = Positive)  

---

## Preprocessing Steps
1. **Lowercasing**: Convert all text to lowercase.  
2. **Removing unwanted symbols/punctuation** while preserving important numbers (e.g., percentages, ratings).  
3. **Tokenization**: Split reviews into individual words using NLTK.  
4. **Stopwords Removal**: Remove common English stopwords to keep meaningful words.  
5. **Lemmatization**: Convert words to their base form using NLTK WordNetLemmatizer.  

After preprocessing, each review is converted to tokens ready for feature extraction.

---
✅ Handled imbalanced data using SMOTE to generate synthetic samples for the minority class, ensuring balanced training data.


## Feature Extraction
- **TF-IDF Vectorization**: Convert the processed text into numerical feature vectors using `TfidfVectorizer` with `max_features=5000`.

---

## Models and Results
| Model | Accuracy |
|-------|---------|
| Naive Bayes | 90.3% |
| Decision Tree Classifier | 87.65% |



