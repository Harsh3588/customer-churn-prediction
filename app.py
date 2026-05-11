import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page Title
st.title("Customer Churn Prediction Dashboard")

# Load Dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Show Dataset
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Churn Distribution
# -----------------------------

st.subheader("Customer Churn Distribution")

fig1, ax1 = plt.subplots()

sns.countplot(x='Churn', data=df, ax=ax1)

st.pyplot(fig1)

# -----------------------------
# Gender Distribution
# -----------------------------

st.subheader("Gender Distribution")

fig2, ax2 = plt.subplots()

df['gender'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    ax=ax2
)

st.pyplot(fig2)

# -----------------------------
# Monthly Charges Distribution
# -----------------------------

st.subheader("Monthly Charges Distribution")

fig3, ax3 = plt.subplots()

sns.histplot(df['MonthlyCharges'], bins=30, ax=ax3)

st.pyplot(fig3)

# -----------------------------
# Contract Type Analysis
# -----------------------------

st.subheader("Contract Type")

fig4, ax4 = plt.subplots()

sns.countplot(x='Contract', data=df, ax=ax4)

plt.xticks(rotation=15)

st.pyplot(fig4)

# -----------------------------
# Correlation Heatmap
# -----------------------------

st.subheader("Correlation Heatmap")

# Copy dataset
df_encoded = df.copy()

# Remove customerID
df_encoded.drop("customerID", axis=1, inplace=True)

# Encode ALL non-numeric columns
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for column in df_encoded.columns:

    if df_encoded[column].dtype == 'object' or str(df_encoded[column].dtype) == 'string':

        df_encoded[column] = le.fit_transform(
            df_encoded[column].astype(str)
        )

# Create heatmap
fig5, ax5 = plt.subplots(figsize=(12,8))

sns.heatmap(
    df_encoded.corr(numeric_only=True),
    cmap='coolwarm',
    ax=ax5
)

st.pyplot(fig5)