
import pandas as pd

data = pd.read_csv("data/Telco-Customer-Churn.csv")

data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"].str.strip(),
    errors="coerce"
)

print("Missing TotalCharges:", data["TotalCharges"].isnull().sum())
print(data["TotalCharges"].dtype)


data = data.dropna(subset=["TotalCharges"])

print("Dataset after cleaning:", data.shape)
print("Missing values:", data["TotalCharges"].isnull().sum())


print("=" * 40)
print("DATASET SHAPE")
print(data.shape)

print("\n" + "=" * 40)
print("COLUMN NAMES")
print(data.columns.tolist())

print("\n" + "=" * 40)
print("MISSING VALUES")
print(data.isnull().sum())

print("\n" + "=" * 40)
print("DUPLICATE ROWS")
print(data.duplicated().sum())

print("\n" + "=" * 40)
print("CHURN DISTRIBUTION")
print(data["Churn"].value_counts())
print(data["Churn"].value_counts(normalize=True) * 100)


print("\nTOTAL CHARGES DATA TYPE")
print(data["TotalCharges"].dtype)

print("\nEMPTY VALUES IN TOTAL CHARGES")
print(data["TotalCharges"].astype(str).str.strip().eq("").sum())

data = data.drop(columns=["customerID"]) 

data["Churn"] = data["Churn"].map({
    "Yes": 1,
    "No": 0
})


# Features: المعلومات اللي الموديل هيتعلم منها
X = data.drop(columns=["Churn"])

# Target: الحاجة اللي عايزين الموديل يتوقعها
y = data["Churn"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Separate numerical and categorical columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object", "str"]).columns

# Prepare the data
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Build the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


from sklearn.metrics import classification_report

# Get churn probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Try a lower threshold
threshold = 0.40
y_pred_tuned = (y_prob >= threshold).astype(int)

print("Logistic Regression - Threshold 0.40")
print(classification_report(y_test, y_pred_tuned))


import joblib

joblib.dump(model, "logistic_regression_model.pkl")

print("Model saved successfully!")