import streamlit as st
import sqlite3
import pandas as pd
import os

st.header("Veri Ekleme")

# Veritabanı klasörü
database_folder = "databases"

# Klasördeki tüm dosyaları al
database_files = [f for f in os.listdir(database_folder) if f.endswith(".db")]

# Seçilen veritabanı
selected_database = st.selectbox("Veritabanı Seçimi:", database_files)

# Veritabanı bağlantısı
conn = sqlite3.connect(f"{database_folder}/{selected_database}")
cursor = conn.cursor()

# Tablo isimlerini al
tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
table_names = [table[0] for table in tables]

# Seçilen tabloyu göster
selected_table = st.selectbox("Tablo Seçimi:", table_names)

# Tabloyu DataFrame'e çevir
query = f"SELECT * FROM {selected_table}"
df = pd.read_sql_query(query, conn)

# Tablonun sütun isimlerini al
column_names = df.columns

# Yeni veri eklemek için bir form oluştur
new_data = {}
for column in column_names:
    new_data[column] = st.text_input(f"{column} Değerini Girin:")

# Kaydet butonu ile yeni veriyi ekleyin
if st.button("Kaydet"):
    values = [new_data[column] for column in column_names]
    cursor.execute(f"INSERT INTO {selected_table} VALUES ({','.join(['?' for _ in column_names])})", values)
    conn.commit()
    st.success("Veri Başarıyla Eklendi!")
    # Yeni veri ekledikten sonra textbox'ları temizle
    for column in column_names:
        new_data[column] = ""

# Alt kısımda tabloyu görüntüle/gizle
show_table = st.checkbox("Tabloyu Görüntüle/Gizle")

if show_table:
    # Alt kısımda tabloyu görüntüle
    st.header("Tablo Görüntüleme")
    st.write(df)

# Veritabanı bağlantısını kapat
conn.close()
