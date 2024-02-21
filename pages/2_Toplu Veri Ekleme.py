import streamlit as st
import sqlite3
import pandas as pd
import os
st.header("Toplu Veri Ekleme")
# Veritabanı klasörü
database_folder = "databases"

# Klasördeki tüm dosyaları al
database_files = [f for f in os.listdir(database_folder) if f.endswith(".db")]

# Var olan bir veritabanı seç
selected_database = st.selectbox("Veritabanı Seçimi:", database_files, index=0)

# Veritabanı bağlantısı
conn = sqlite3.connect(f"{database_folder}/{selected_database}")
cursor = conn.cursor()

# Var olan bir tablo seç
existing_tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
existing_table_names = [table[0] for table in existing_tables]

# Tablo seçimi
selected_table = st.selectbox("Tablo Seçimi:", existing_table_names, index=0)

# CSV dosyası yükleme
uploaded_file = st.file_uploader("CSV Dosyasını Yükleyin", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')

    # Veritabanındaki sütunları ve CSV sütunlarını ekrana yazdır
    st.header("Veritabanı Sütunları:")
    db_columns = [db_column[1] for db_column in cursor.execute(f"PRAGMA table_info({selected_table})").fetchall()]
    st.write(db_columns)

    st.header("CSV Sütunları:")
    st.write(df.columns)

    # Sütunları eşleştirme
    column_mapping = {}
    for db_column in db_columns:
        selected_csv_column = st.selectbox(f"{db_column} için CSV Sütunu Seçimi:",
                                           ["(Seçim Yapın)"] + list(df.columns))
        column_mapping[db_column] = selected_csv_column

    # Veritabanına verileri ekle
    if st.button("Verileri Veritabanına Ekle"):
        for index, row in df.iterrows():
            values = [row[column_mapping[db_column]] for db_column in db_columns]
            # None olmayan değerleri ekleyelim
            filtered_values = [value for value in values if value is not None]
            if filtered_values:
                cursor.execute(f"INSERT INTO {selected_table} ({','.join(db_columns)}) VALUES ({','.join(['?' for _ in filtered_values])})", filtered_values)
        conn.commit()
        st.success("Veriler Başarıyla Eklendi!")

# Alt kısımda tabloyu görüntüle/gizle
show_table = st.checkbox("Tabloyu Görüntüle/Gizle")

if show_table:
    # Alt kısımda tabloyu görüntüle
    st.header("Tablo Görüntüleme")
    df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
    st.write(df)

# Veritabanı bağlantısını kapat
conn.close()
