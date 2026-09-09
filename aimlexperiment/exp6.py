
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

                        #Manya Nayak 500123429

# --------------------------------------------------
# 1. Check file path
# --------------------------------------------------
print("Current working directory:", os.getcwd())

file_path = "spam.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"'{file_path}' not found. "
        "Place spam.csv in the same folder as this Python file."
    )

print("spam.csv found! Loading dataset...\n")

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------
data = pd.read_csv(file_path, encoding="latin-1")

# Keep only the first two columns
data = data.iloc[:, :2]
data.columns = ["label", "message"]

# Remove missing values and duplicates
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Convert labels: ham = 0, spam = 1
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})
                              #Manya Nayak 500123429
# Remove rows with invalid labels
data.dropna(inplace=True)

print("Dataset loaded successfully!")
print("Total messages:", len(data))
print("\nFirst 5 rows:")
print(data.head())

# --------------------------------------------------
# 3. Split data into training and testing
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["message"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# --------------------------------------------------
# 4. Convert text into TF-IDF features
# --------------------------------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --------------------------------------------------
# 5. Train Naive Bayes model
# --------------------------------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
                       #Manya Nayak 500123429
# --------------------------------------------------
# 6. Make predictions
# --------------------------------------------------
y_pred = model.predict(X_test_tfidf)

# --------------------------------------------------
# 7. Evaluate model
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 40)
print("MODEL EVALUATION")
print("=" * 40)

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# 8. Test custom messages
# --------------------------------------------------
emails = [
    "Congratulations! You have won a free lottery prize. Claim now!",
    "Hey, can you send me today's assignment?",
    "URGENT! You have won $5000. Click the link to claim your reward.",
    "Let's meet for lunch tomorrow."
]
                #Manya Nayak 500123429
email_features = vectorizer.transform(emails)
predictions = model.predict(email_features)

print("\n" + "=" * 40)
print("CUSTOM MESSAGE PREDICTIONS")
print("=" * 40)

for email, prediction in zip(emails, predictions):
    if prediction == 1:
        print("SPAM:", email)
    else:
        print("HAM:", email)