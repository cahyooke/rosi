import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from ucimlrepo import fetch_ucirepo

# Load and preprocess data
@st.cache_data
def load_data():
    wholesale_customers = fetch_ucirepo(id=292)
    data = wholesale_customers.data.original.copy()

    # Encode all columns
    le = LabelEncoder()
    for col in data.columns:
        data[col] = le.fit_transform(data[col])

    # Impute missing values with KNNImputer
    imputer = KNNImputer(n_neighbors=9, metric='nan_euclidean')
    data_imputed = imputer.fit_transform(data)
    data_imputed_df = pd.DataFrame(data_imputed, columns=data.columns)

    return data_imputed_df

data = load_data()

# Feature selection
feature_columns = ['Channel', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = data[feature_columns]
y = data['Region']

# Split and train model with best k=5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Prediksi Region Wholesale Customers")
st.markdown("Masukkan fitur-fitur berikut untuk memprediksi Region:")

channel = st.number_input("Channel", min_value=0, step=1)
fresh = st.number_input("Fresh", min_value=0)
milk = st.number_input("Milk", min_value=0)
grocery = st.number_input("Grocery", min_value=0)
frozen = st.number_input("Frozen", min_value=0)
detergents = st.number_input("Detergents_Paper", min_value=0)
delicassen = st.number_input("Delicassen", min_value=0)

if st.button("Prediksi"):
    user_data = pd.DataFrame([[channel, fresh, milk, grocery, frozen, detergents, delicassen]],
                              columns=feature_columns)
    prediction = model.predict(user_data)[0]
    st.success(f"Region yang diprediksi: {prediction}")
