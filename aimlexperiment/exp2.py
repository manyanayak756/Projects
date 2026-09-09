import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#manya nayak 500123429

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)


# Train model
model.fit(X_train, y_train)


# Predict species
y_pred = model.predict(X_test)

# Manya Nayak 500123429
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", round(accuracy * 100, 2), "%")


# Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print(
    "Predicted Species:",
    iris.target_names[prediction[0]]
)


# Visualization
plt.scatter(
    X_test[:, 0],
    X_test[:, 2],
    c=y_test
)

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Iris Flower Classification")

plt.show()