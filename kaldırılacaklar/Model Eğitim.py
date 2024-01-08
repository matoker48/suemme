import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from itertools import combinations
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import time
import streamlit as st
import os
import sqlite3
#---------------------------------------------------------------------------
def calculate_r_squared(y, y_pred):
    ss_total = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - y_pred) ** 2).sum()
    return 1 - (ss_res / ss_total)
#---------------------------------------------------------------------------
def calculate_p_values(model, X, y):
    n = len(y)
    p = X.shape[1]  # Number of features
    dof = n - p - 1  # Degrees of freedom

    # Residual standard error
    ss_res = ((y - model.predict(X)) ** 2).sum()
    mse = ss_res / dof
    se = np.sqrt(np.diagonal(mse * np.linalg.inv(np.dot(X.T, X))))

    # Calculate t-statistic and p-values
    t = model.coef_ / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t), dof))

    return p_values
#---------------------------------------------------------------------------
def standardize(X):
    # Standart scaller'ı oluştur
    scaler = StandardScaler()
    # Standartlaştır
    standardized_X = scaler.fit_transform(X)
    # DataFrame'e çevir
    standardized_X = pd.DataFrame(standardized_X, columns=X.columns)
    return standardized_X
#------------------------------------------------------------------------------------------    
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data
#--------------------------------------------------------------------------------------------------

def find_best_polynomial_combinations(X, y,max_degree , top_n, max_terms):
    poly = PolynomialFeatures(degree=max_degree)
    X_poly = poly.fit_transform(X)

    feature_names = poly.get_feature_names_out(input_features=X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names)

    included_indices = list(range(X_poly.shape[1]))

    all_combinations = []

    for r in range(2, max_terms + 1):
        combinations_at_level = list(combinations(included_indices, r))
        for combo in combinations_at_level:
            terms = []

            for j in range(len(combo)):
                feature_index = included_indices[combo[j]]
                term = feature_names[feature_index]
                terms.append(term)

            model = LinearRegression()
            # Modify indexing to select relevant columns
            model.fit(X_poly_df[terms], y)
            y_pred = model.predict(X_poly_df[terms])
            r2 = calculate_r_squared(y, y_pred)
            p_values = calculate_p_values(model, X_poly_df[terms], y)
            feature_p_values = dict(zip(terms, p_values))

            vif_data = calculate_vif(X_poly_df[terms])
            vif_values = dict(zip(vif_data["Variable"], vif_data["VIF"]))

            if all(p <= 0.05 for p in p_values) and all(p <=5  for p in vif_data["VIF"]):
                all_combinations.append((combo, r2, terms, vif_values,feature_p_values))

    all_combinations.sort(key=lambda x: x[1], reverse=True)
    top_combinations = all_combinations[:top_n]

    return top_combinations
# Streamlit uygulaması
def main():
    st.title("Polinomiyal Regresyon Model Kombinasyonu")

    # Veritabanı seçimi
    databases_folder = "databases"  # Veritabanı klasörünün adı
    database_names = [name.split(".")[0] for name in os.listdir(databases_folder) if name.endswith(".db")]
    selected_database = st.selectbox("Veri Tabanı Seç", database_names)

    # Seçilen veritabanının tablolarını al
    database_path = os.path.join(databases_folder, f"{selected_database}.db")
    connection = sqlite3.connect(database_path)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", con=connection)
    connection.close()
    table_names = tables["name"].tolist()
    selected_table = st.selectbox("Tablo Seç", table_names)

    # Veritabanındaki sütun isimlerini çek
    connection = sqlite3.connect(database_path)
    query = f"PRAGMA table_info({selected_table});"
    columns = pd.read_sql_query(query, con=connection)
    connection.close()

    # Bağımsız ve bağımlı değişken seçimi
    st.subheader("Değişken Seçimi")
    dependent_var = st.selectbox("Hedef Değişkeni Seç", columns['name'].tolist())
    independent_vars = st.multiselect("Bağımsız Değişkenleri Seç", columns['name'].tolist())

    # Verileri yükle
    connection = sqlite3.connect(database_path)

    # Bağımsız değişkenler seçilmişse SQL sorgusunu oluştur, aksi halde hata önle
    if independent_vars:
        df = pd.read_sql_query(f"SELECT {', '.join(independent_vars)}, {dependent_var} FROM {selected_table};", con=connection)
    else:
        st.error("Bağımsız değişkenleri seçmediniz. Lütfen en az iki bağımsız değişken seçin.")
        st.stop()

    connection.close()
    X = df[independent_vars]
    X = standardize(X)
    y = df[dependent_var]
    n = len(y)

    # Polynomial Regression Analysis
    st.header("En İyi Kombinasyonlar")

    # Polynomial derecesi seçimi
    max_degree = st.slider("Polinom Derecesi Seç", min_value=1, max_value=5, value=2)

    # Measure the start time
    start_time = time.time()
    top_10_combinations = find_best_polynomial_combinations(X, y,max_degree , top_n=1, max_terms=20)
    # Measure the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Print the elapsed time
    st.write(f"Elapsed Time: {elapsed_time} seconds")
    for i, (combo, r2, terms, vif_values, feature_p_values) in enumerate(top_10_combinations):
        st.subheader(f"Combination {i + 1}")
        st.write("--------------------------")
        st.write(f"Combo: {combo}")
        st.write("--------------------------")
        st.write(f"Terms: {terms}")
        st.write("--------------------------")
        st.write("VIF Values:")
        st.write(vif_values)
        st.write("--------------------------")
        st.write("Feature P-Values:")
        st.write(feature_p_values)
        st.write("--------------------------")
        st.write(f"R-squared: {r2}")
        st.write("--------------------------")


# Uygulamayı çalıştır
if __name__ == "__main__":
    main()