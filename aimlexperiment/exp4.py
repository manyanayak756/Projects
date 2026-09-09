from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

#Manya Nayak
#500123429


data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df["Target"] = data.target

print(df.head()) 

# Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression(max_iter=10000)

# Train model
model.fit(X_train, y_train)

#Manya Nayak
#500123429

# Predict
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")
# Predict one sample
prediction = model.predict([X_test[0]])

print("Predicted:", data.target_names[prediction[0]])
print("Actual:", data.target_names[y_test[0]])