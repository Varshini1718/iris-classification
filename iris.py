import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import os
import joblib

df = pd.read_csv("IRIS.csv")

df.head()
print(df.shape)
print(df.info())
print(df.describe())

df.columns = df.columns.str.strip()
df.isnull().sum()

id_col = next((c for c in df.columns if c.lower() == "id"), None)
if id_col is not None:
    df.drop(id_col, axis=1, inplace=True)

species_col = next((c for c in df.columns if c.lower() == "species"), None)
if species_col is None:
    raise KeyError("No species column found in CSV. Expected 'species' or 'Species'.")

sns.pairplot(df, hue=species_col)
plt.show()
plt.figure(figsize=(8,6))

sns.heatmap(
    df.drop(species_col, axis=1).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")
print(plt.show())
X = df.drop(species_col, axis=1)
y = df[species_col]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)
knn_preds = knn.predict(X_test_scaled)

lr = LogisticRegression(max_iter=200)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)
dt_preds = dt.predict(X_test_scaled)

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name}")
    print("-" * 40)

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return acc
knn_acc = evaluate_model("k-NN", y_test, knn_preds)

lr_acc = evaluate_model("Logistic Regression", y_test, lr_preds)

dt_acc = evaluate_model("Decision Tree", y_test, dt_preds)
results = {
    "k-NN": knn_acc,
    "Logistic Regression": lr_acc,
    "Decision Tree": dt_acc
}

best_model_name = max(results, key=results.get)

print("Best Model:", best_model_name)

best_model = {
    "k-NN": knn,
    "Logistic Regression": lr,
    "Decision Tree": dt
}[best_model_name]

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_iris_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Load model & scaler
model = joblib.load("models/best_iris_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Example flower
sample = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

# Scale
sample_scaled = scaler.transform(sample)

# Predict
prediction = model.predict(sample_scaled)

print("Predicted Species:", prediction[0])