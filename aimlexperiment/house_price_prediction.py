import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Manya Nayak
# SAP ID: 500123429

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data.csv")

# =========================
# EDA
# =========================

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

# Remove invalid prices
df = df[df["price"] > 0]

# =========================
# PRICE DISTRIBUTION
# =========================

plt.figure(figsize=(8, 5))
plt.hist(df["price"], bins=50)

plt.xlabel("House Price")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Prices")

plt.show()

# =========================
# LIVING AREA VS PRICE
# =========================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["sqft_living"],
    df["price"],
    alpha=0.5
)

plt.xlabel("Living Area (sqft)")
plt.ylabel("House Price")
plt.title("Living Area vs House Price")

plt.show()

# =========================
# BEDROOMS VS PRICE
# =========================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["bedrooms"],
    df["price"],
    alpha=0.5
)

plt.xlabel("Number of Bedrooms")
plt.ylabel("House Price")
plt.title("Bedrooms vs House Price")

plt.show()

# =========================
# BATHROOMS VS PRICE
# =========================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["bathrooms"],
    df["price"],
    alpha=0.5
)

plt.xlabel("Number of Bathrooms")
plt.ylabel("House Price")
plt.title("Bathrooms vs House Price")

plt.show()

# =========================
# CORRELATION HEATMAP
# =========================

plt.figure(figsize=(12, 8))

correlation = df.select_dtypes(
    include="number"
).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.show()

# =========================
# FEATURES
# =========================

features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "city",
    "statezip"
]

X = df[features]
y = df["price"]

# =========================
# PREPROCESSING
# =========================

categorical = ["city", "statezip"]

preprocessor = ColumnTransformer(
    [
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        )
    ],
    remainder="passthrough"
)

# =========================
# MODEL
# =========================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression", LinearRegression())
])

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=50
)

# =========================
# TRAIN MODEL
# =========================

model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================

y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("------------------")
print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 3))

# =========================
# ACTUAL VS PREDICTED
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

# Perfect prediction line
minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.text(
    minimum,
    maximum * 0.9,
    f"MAE = {mae:.2f}\nR² = {r2:.3f}"
)

plt.show()