import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/aksha/OneDrive/Desktop/project/diabetes.csv")
    df.fillna(df.median(numeric_only=True), inplace=True)
    return df

df = load_data()
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train and compare models
@st.cache_data
def train_models():
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier()
    }
    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        accuracies[name] = acc
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    return accuracies, best_model_name, best_model

accuracies, best_model_name, best_model = train_models()

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["📊 Visualizations", "📈 Model Comparison", "🩺 Predict Diabetes"])

# ----------------------
# Page 1: Data Visualizations
# ----------------------
if page == "📊 Visualizations":
    st.title("Data Visualizations")
    st.write("Click a button to display a specific chart:")

    if st.button("Show BMI Distribution"):
        fig, ax = plt.subplots()
        ax.hist(df['BMI'], bins=20, color='orange')
        ax.set_title('BMI Distribution')
        st.pyplot(fig)

    if st.button("Show Glucose Distribution"):
        fig, ax = plt.subplots()
        ax.hist(df['Glucose'], bins=20, color='green')
        ax.set_title('Glucose Distribution')
        st.pyplot(fig)

    if st.button("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='viridis', ax=ax)
        st.pyplot(fig)

# ----------------------
# Page 2: Model Comparison
# ----------------------
elif page == "📈 Model Comparison":
    st.title("Model Accuracy Comparison")
    st.write("Accuracy of different models on the diabetes dataset:")

    acc_df = pd.DataFrame(list(accuracies.items()), columns=["Model", "Accuracy"])
    st.dataframe(acc_df)

    fig, ax = plt.subplots()
    sns.barplot(x="Model", y="Accuracy", data=acc_df, palette="Blues_d", ax=ax)
    ax.set_ylim(0.6, 0.8)  # Adjusted to match actual accuracy range
    ax.set_title("Model Accuracy Comparison")
    st.pyplot(fig)

    st.success(f"Best Model: {best_model_name} ({accuracies[best_model_name]*100:.2f}% Accuracy)")

# ----------------------
# Page 3: Prediction Interface
# ----------------------
elif page == "🩺 Predict Diabetes":
    st.title("Diabetes Prediction")
    st.write("Enter patient health information:")

    preg = st.slider("Pregnancies", 0, 20, 1)
    glucose = st.slider("Glucose", 0, 300, 110)
    bp = st.slider("Blood Pressure", 0, 140, 70)
    skin = st.slider("Skin Thickness", 0, 99, 20)
    insulin = st.slider("Insulin", 0, 846, 79)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Age", 10, 100, 33)

    user_data = pd.DataFrame([[preg, glucose, bp, skin, insulin, bmi, dpf, age]],
                              columns=X.columns)
    user_scaled = scaler.transform(user_data)

    if st.button("Predict"):
        prediction = best_model.predict(user_scaled)[0]
        prob = best_model.predict_proba(user_scaled)[0][1]

        if prediction == 1:
            st.error(f"🔴 The patient is likely Diabetic. (Confidence: {prob:.2%})")
        else:
            st.success(f"🟢 The patient is Not Diabetic. (Confidence: {1-prob:.2%})")
