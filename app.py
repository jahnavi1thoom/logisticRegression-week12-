import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------- PAGE CONFIG ----------------
st.set_page_config("Customer Churn Prediction", layout="centered")

st.title("📊 Telecom Customer Churn Prediction")

# ---------------- LOAD DATA ----------------

st.subheader("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload Telco Customer Churn CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()
@st.cache_data
def load_df(file):
    return pd.read_csv(file)

df = load_df(uploaded_file)




# ---------------- DATA CLEANING ----------------
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------------- FEATURE / TARGET ----------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# ---------------- PREPROCESSING PIPELINE ----------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000))
])

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- TRAIN MODEL ----------------
model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# ---------------- METRICS ----------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# ---------------- DATA PREVIEW ----------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.subheader("📊 Confusion Matrix (Diagram)")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Predicted No Churn", "Predicted Churn"],
    yticklabels=["Actual No Churn", "Actual Churn"],
    ax=ax
)

ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)




# ---------------- MODEL PERFORMANCE ----------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Churn Rate", f"{df['Churn'].mean():.2f}")

st.text("Confusion Matrix")
st.write(cm)

st.text("Classification Report")
st.text(classification_report(y_test, y_pred))

# ---------------- PREDICTION UI ----------------
st.subheader("🔮 Predict Customer Churn")

tenure = st.slider("Tenure (months)", 0, int(df["tenure"].max()), 12)
monthly = st.slider("Monthly Charges", 20, int(df["MonthlyCharges"].max()), 70)
contract = st.selectbox("Contract", df["Contract"].unique())
internet = st.selectbox("Internet Service", df["InternetService"].unique())
payment = st.selectbox("Payment Method", df["PaymentMethod"].unique())

input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly,
    "Contract": contract,
    "InternetService": internet,
    "PaymentMethod": payment
}])

# fill remaining columns with most common values
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = df[col].mode()[0]

input_df = input_df[X.columns]

prob = model.predict_proba(input_df)[0][1]
prediction = "Likely to Churn" if prob >= 0.5 else "Likely to Stay"

st.success(f"**Prediction:** {prediction}")
st.info(f"**Churn Probability:** {prob:.2%}")
