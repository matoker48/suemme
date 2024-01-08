import streamlit as st
import os
import sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
import bestmodel as bm
from sklearn.preprocessing import StandardScaler


# "Su Emme Tahmini" başlığını ekleyin
st.title("Su Emme Tahmini")
# Veritabanı dosyalarının bulunduğu klasörü belirtin
database_folder = "databases"  # Bu klasörde veritabanı dosyalarınız bulunsun

# Klasördeki veritabanı dosyalarını alın
database_files = [f for f in os.listdir(database_folder) if f.endswith(".db")]

# Klasörde veritabanı dosyası yoksa uyarı verin
if not database_files:
    st.warning("Klasörde hiç veritabanı dosyası bulunamadı.")
else:
    # ComboBox ile veritabanı seç
    selected_database = st.selectbox("Veritabanı Seç", database_files)

    # Seçilen veritabanı adını göster
    st.write(f"Seçilen Veritabanı: {selected_database}")

    # Seçilen veritabanına bağlan
    selected_database_path = os.path.join(database_folder, selected_database)
    conn = sqlite3.connect(selected_database_path)
    cursor = conn.cursor()

    # Tabloları çek
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    # ComboBox ile tablo seç
    selected_table = st.selectbox("Tablo Seç", tables)

    # Seçilen tablonun sütunlarını çek
    cursor.execute(f"PRAGMA table_info({selected_table});")
    columns = [row[1] for row in cursor.fetchall()]

    # Birinci sütunu seç
    selected_column1 = st.selectbox("Bağımlı Değişkeni Seç", columns)

    # İkinci sütunu çoklu seçim yapılabilen DropDownList ile seç
    selected_column2 = st.multiselect("Bağımsız Değişkeni Seç", [col for col in columns if col != selected_column1])

    # Veritabanı bağlantısını kapat
    conn.close()
# Verileri çekip birinci DataFrame'e kaydedelim
df1 = None
if selected_database and selected_table:
    conn = sqlite3.connect(selected_database_path)
    query = f"SELECT {selected_column1} FROM {selected_table}"
    df1 = pd.read_sql_query(query, conn)
    conn.close()

# İkinci DataFrame'i kaydedelim
df2 = None
if selected_database and selected_table and selected_column2:
    conn = sqlite3.connect(selected_database_path)
    query = f"SELECT {', '.join(selected_column2)} FROM {selected_table}"
    df2 = pd.read_sql_query(query, conn)
    conn.close()
# combobox için değerleri tanımla
items = ["Regresyon Analizi", "Decision Tree", "Random Forest", "Support Vector Machine"]

# combobox'u oluştur
model_type = st.selectbox("Model Türü", items)


# seçilen model türünü yazdır
st.write("Seçilen model türü:", model_type)


# DataFrame'leri birleştirip yazdır
if df1 is not None and df2 is not None:
    df = pd.concat([df1, df2], axis=1)
    st.dataframe(df)
else:
    st.error("Please select both databases and tables, and at least one column for the second DataFrame.")

def simple_linear_regression(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    y_pred = model.predict(x)
    return model, y_pred
def standardize(X):

    # Standart scaller'ı oluştur
    scaler = StandardScaler()

    # Standartlaştır
    standardized_X = scaler.fit_transform(X)

    # DataFrame'e çevir
    standardized_X = pd.DataFrame(standardized_X, columns=X.columns)

    return standardized_X
# "Regresyon Analizi Çalıştır" butonunu oluştur
button_clicked = st.button("Regresyon Analizi Çalıştır")

# Butona tıklandığında "simple_linear_regression()" fonksiyonunu çağır
if button_clicked:
    # seçilen model türüne göre fonksiyonu çalıştır
    if model_type == "Regresyon Analizi":

        model, y_pred = simple_linear_regression(df2, df1)
        st.title("Regresyon Analizi Özeti")
        st.write(model.summary())
        result_df = pd.DataFrame({'Gerçek Değer (y)': df1.squeeze(), 'Tahmin Edilen Değer (y_pred)': y_pred})
        st.subheader("Gerçek ve Tahmin Edilen Değerler")
        df = pd.concat([df1, y_pred.rename('y_pred')], axis=1)
        st.dataframe(df)
        #------------------------------------------------------------------
        st.title("En İyi Modeller")
        df2 = standardize(df2)
        results = bm.find_best_polynomial_combinations(df2, df1,top_n=5, max_degree=3, max_terms=15)
        for i, (combo, r2, terms,vif_values,feature_p_values) in enumerate(results):
            st.write("--------------------------")
            st.write(combo)
            st.write("--------------------------")
            st.write(terms)
            for feature, p_value in vif_values.items():
                st.write(f"{feature}: {p_value}")
            st.write("--------------------------")
            for feature, p_value in feature_p_values.items():
                st.write(f"{feature}: {p_value}")
            st.write("--------------------------")
            st.write("R-squared:", r2)
            st.write("--------------------------")
            st.write("##################################################")
    elif model_type == "Decision Tree":
        pass
    elif model_type == "Random Forest":
        pass
    elif model_type == "Support Vector Machine":
        pass


