import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a small dataset
data = {
    "review": [
        "I loved this movie",
        "This movie was excellent",
        "Amazing experience",
        "Very good product",
        "I really enjoyed it",
        "Fantastic and wonderful",
        "The product is great",
        "I am very happy with this",
        "Absolutely loved it",
        "Best product ever",
        "I hated this movie",
        "This movie was terrible",
        "Very bad experience",
        "Worst product ever",
        "I did not like it",
        "The product is horrible",
        "Very disappointing",
        "I am unhappy with this",
        "Waste of money",
        "Completely useless"
    ],
    "sentiment": [
        "positive", "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative", "negative"
    ]
}

df = pd.DataFrame(data)

# Step 2: Separate input and output
X = df["review"]
y = df["sentiment"]

# Step 3: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Convert text into numerical values using TF-IDF
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)                        # what is the significance of vectorizer.fit_transform(X_train) in this code?
X_test = vectorizer.transform(X_test)

# Step 5: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)                                          #why logistic regression is used here

# Step 6: Test the model
y_pred = model.predict(X_test)

# Step 7: Display accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Take a new review from the user
review = input("\nEnter a review: ")

review_vector = vectorizer.transform([review])
prediction = model.predict(review_vector)

print("Sentiment:", prediction[0])