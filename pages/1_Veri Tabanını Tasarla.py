import streamlit as st
import sqlite3
import pandas as pd
import os
st.header("Veri Tabanı Oluştur")

# Veritabanı klasörü
database_folder = "databases"

# Klasördeki tüm dosyaları al
database_files = [f for f in os.listdir(database_folder) if f.endswith(".db")]

# Veritabanı seçimi veya oluşturma için radio butonları
selection_mode = st.radio("Veritabanı Seçimi veya Oluştur:", ["Veritabanı Seç", "Veritabanı Oluştur"])

if selection_mode == "Veritabanı Oluştur":
    new_database_name = st.text_input("Yeni Veritabanı Adı:")
    if st.button("Veritabanı Oluştur") and new_database_name:
        new_database_path = os.path.join(database_folder, f"{new_database_name}.db")
        conn = sqlite3.connect(new_database_path)
        conn.close()
        st.success(f"{new_database_name} Veritabanı Başarıyla Oluşturuldu!")
        database_files.append(f"{new_database_name}.db")

# Var olan bir veritabanı seç
if not database_files:
    st.warning("Henüz bir veritabanı oluşturulmamış. Lütfen önce yeni bir veritabanı oluşturun.")
else:
    selected_database = st.selectbox("Veritabanı Seçimi:", database_files, index=0)

    # Veritabanı bağlantısı
    conn = sqlite3.connect(f"{database_folder}/{selected_database}")
    cursor = conn.cursor()
# Var olan bir tablo seç
existing_tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
existing_table_names = [table[0] for table in existing_tables]

# Tablo seçimi
selected_table = st.selectbox("Tablo Seçimi:", existing_table_names, index=0)

# İşlem seçimi
actions = ["Yeni Tablo Oluştur", "Tablo Sil", "Sütun Ekle", "Sütun Sil"]
action = st.radio("İşlem Seçimi:", actions)

if action == "Yeni Tablo Oluştur":
    # Yeni tablo oluşturmak için bir form oluştur
    st.header("Yeni Tablo Oluşturma")
    new_table_name = st.text_input("Yeni Tablo Adı:")
    if st.button("Tablo Oluştur") and new_table_name:
        cursor.execute(f"CREATE TABLE {new_table_name} (id REAL, name TEXT);")
        conn.commit()
        st.success(f"{new_table_name} Tablosu Başarıyla Oluşturuldu!")
        # Yeni tablo oluşturulduktan sonra tabloyu güncelle
        existing_tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        existing_table_names = [table[0] for table in existing_tables]
        selected_table = new_table_name

elif action == "Tablo Sil":
    # Var olan tabloları seçme
    table_to_delete = st.selectbox("Silinecek Tabloyu Seçin:", existing_table_names)
    if st.button("Tabloyu Sil"):
        cursor.execute(f"DROP TABLE IF EXISTS {table_to_delete}")
        conn.commit()
        st.success(f"{table_to_delete} Tablosu Başarıyla Silindi!")
        # Tablo silindikten sonra tabloyu güncelle
        existing_tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        existing_table_names = [table[0] for table in existing_tables]
        selected_table = existing_table_names[0] if existing_table_names else ""

elif action == "Sütun Ekle":
    # Yeni sütun ekleme için bir form oluştur
    st.header("Yeni Sütun Ekleme")
    new_column_name = st.text_input("Yeni Sütun Adı:")
    if st.button("Sütun Ekle") and new_column_name:
        cursor.execute(f"ALTER TABLE {selected_table} ADD COLUMN {new_column_name} REAL;")
        conn.commit()
        st.success(f"{new_column_name} Sütunu Başarıyla Eklendi!")

elif action == "Sütun Sil":
    # Var olan sütunları seçme
    existing_columns = cursor.execute(f"PRAGMA table_info({selected_table})").fetchall()
    existing_column_names = [col[1] for col in existing_columns]

    # Silinecek sütunu seçme
    selected_column = st.selectbox("Silinecek Sütun Seçimi:", existing_column_names)

    if st.button("Sütun Sil"):
        cursor.execute(f"ALTER TABLE {selected_table} DROP COLUMN {selected_column};")
        conn.commit()
        st.success(f"{selected_column} Sütunu Başarıyla Silindi!")

# Alt kısımda tabloyu görüntüle/gizle
if st.checkbox("Tabloyu Görüntüle/Gizle"):
    # Alt kısımda tabloyu görüntüle
    st.header("Tablo Görüntüleme")
    df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
    st.write(df)

# Veritabanı bağlantısını kapat
conn.close()
