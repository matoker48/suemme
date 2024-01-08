import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def standardize(X):

    # Standart scaller'ı oluştur
    scaler = StandardScaler()

    # Standartlaştır
    standardized_X = scaler.fit_transform(X)

    # DataFrame'e çevir
    standardized_X = pd.DataFrame(standardized_X, columns=X.columns)

    return standardized_X
def generate_polynomial_model(X, y, indices, degree):
    # Modelin indekslerine göre terimleri belirleyin
    terms = [f'x{i}' for i in indices]

    # Polynomial özellikleri ekleyin
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    # Model terimlerini seçin
    selected_terms = np.array(terms)

    # Veriyi eğitim ve test setlerine böl
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

    # Modeli oluşturun
    model = LinearRegression()

    # Modeli eğitin
    model.fit(X_train, y_train)

    # Tahminleri alın
    y_pred = model.predict(X_test)

    # R2 skoru hesapla
    r2 = r2_score(y_test, y_pred)

    # MAE hesapla
    mae = mean_absolute_error(y_test, y_pred)

    return selected_terms, r2, mae


# Veritabanı seçimi
databases_folder = "databases"
database_names = [name.split(".")[0] for name in os.listdir(databases_folder) if name.endswith(".db")]
selected_database = st.selectbox("Select a database", database_names)

# Seçilen veritabanının tablolarını al
database_path = os.path.join(databases_folder, f"{selected_database}.db")
connection = sqlite3.connect(database_path)
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", con=connection)
connection.close()
table_names = tables["name"].tolist()
selected_table = st.selectbox("Select a table", table_names)

# Veritabanındaki sütun isimlerini çek
connection = sqlite3.connect(database_path)
query = f"PRAGMA table_info({selected_table});"
columns = pd.read_sql_query(query, con=connection)
connection.close()

# Bağımsız ve bağımlı değişken seçimi
st.subheader("Variable Selection")
independent_vars = st.multiselect("Select independent variables", columns['name'].tolist())
dependent_var = st.selectbox("Select dependent variable", columns['name'].tolist())

# Verileri yükle
connection = sqlite3.connect(database_path)
df = pd.read_sql_query(f"SELECT {', '.join(independent_vars)}, {dependent_var} FROM {selected_table};", con=connection)
connection.close()

# Standardize the independent variables
X = df[independent_vars]
X = standardize(X)
y = df[dependent_var]

# Polynomial Regression Analysis
st.header("Polynomial Regression Analysis Results")

# Polynomial degree selection
degree = st.slider("Select polynomial degree", min_value=1, max_value=5, value=2)

indices = [2, 3, 6, 8, 9] 
# Generate Polynomial Model
selected_terms, r2_score, mae_score = generate_polynomial_model(X, y, indices, degree)

# Display results
st.write(f"Selected Terms: {selected_terms}")
st.write(f"R2 Score: {r2_score}")
st.write(f"MAE (Mean Absolute Error): {mae_score}")
