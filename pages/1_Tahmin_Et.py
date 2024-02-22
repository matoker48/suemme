import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
import streamlit as st
import os
import sqlite3
import regresyon as reg  # Import your regresyon module here
import ml  # Import your machine learning model here
from sklearn.model_selection import GridSearchCV, train_test_split,cross_val_score,StratifiedKFold
from sklearn.metrics import make_scorer,r2_score, mean_absolute_error
import pickle
import json
import datetime


def load_data_from_csv(file):
    data = pd.read_csv(file, sep=';')
    return data

def load_data_from_excel(file):
    data = pd.read_excel(file)
    return data

def load_data_from_database(database_path, table_name):
    conn = sqlite3.connect(database_path)
    data = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
    conn.close()
    return data

def standardize_data(X_train, df_s):
    scaler = StandardScaler()
    standardized_X_train = scaler.fit_transform(X_train)
    standardized_df_s = scaler.transform(df_s)
    return pd.DataFrame(standardized_X_train, columns=X_train.columns),  pd.DataFrame(standardized_df_s, columns=df_s.columns)

def select_features_and_dependent_variable(data):
    dependent_variable = st.selectbox("Bağımlı Değişken:", data.columns)
    independent_variables = [col for col in data.columns if col != dependent_variable]
    selected_independent_variables = st.multiselect("Bağımsız Değişkenleri Seçin:", independent_variables)
    return dependent_variable, selected_independent_variables

def get_user_input(selected_independent_variables):
    user_input = {}
    for col in selected_independent_variables:
        value = st.number_input(f"{col} Değerini Girin:")
        user_input[col] = [value]
    return pd.DataFrame(user_input)

def get_model_info(model_adi):
    """
    Veritabanından model bilgilerini alır ve ilgili değerleri döndürür.

    Parametreler:
        model_adi: Modelin adı.

    Dönüş Değeri:
        Bağımlı değişken, bağımsız değişkenler, model tipi, dosya adı
    """

    # Veritabanı bağlantısı
    database_path = f"models/{model_adi}"
    conn = sqlite3.connect(database_path)
    query = f"SELECT * FROM model_info;"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data


def load_model_from_records(model_name):
    """
    Belirtilen model adına göre records klasöründen modeli yükler.

    Parametreler:
        model_name: Yüklenmek istenen modelin adı.

    Dönüş Değeri:
        Yüklenen model.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ust_dizin_yolu = os.path.abspath(os.path.join(current_dir, ".."))

    model_path = os.path.join(ust_dizin_yolu,"records", f"{model_name}")
    print("-----------------------------------------------------------")
    print(model_path)

    try:
        with open(model_path, "rb") as file:
            loaded_model = pickle.load(file)
            return loaded_model
    except FileNotFoundError:
        return None

def evaluate_model_with_cross_validation(model, X, y,df, cv=5):
    Xs, dfs = standardize_data(X, df)
    print(X)
    loaded_model = load_model_from_records(model)
    if loaded_model is None:
        print("sdadfsfasf")
    # R^2 skoru için scorer oluştur
    r2_scorer = make_scorer(r2_score)
    # MAE skoru için scorer oluştur
    mae_scorer = make_scorer(mean_absolute_error)

    
    skf = StratifiedKFold(n_splits=cv)

    # Cross-validation ile R^2 ve MAE skorlarını hesapla
    r2_scores = cross_val_score(loaded_model, Xs, y, cv=cv, scoring=r2_scorer)
    mae_scores = cross_val_score(loaded_model, Xs, y, cv=cv, scoring=mae_scorer)
    mae = mean_absolute_error(y, loaded_model.predict(Xs))
    r2 = r2_score(y, loaded_model.predict(Xs))
    
    # Ortalama skorları al
    mean_r2_score = np.mean(r2_scores)
    mean_mae_score = np.mean(mae_scores)

    # Modelin tahmin yapma yeteneğini değerlendir
    predictions = loaded_model.predict(Xs)
    predictions[predictions < 0] = 0
    input_pred = loaded_model.predict(dfs)
    input_pred[input_pred < 0] = 0

    return mean_r2_score, mean_mae_score, predictions, input_pred

def evaluate_model_with_cross_validation_for_reg(model, X, y, df, indices, cv=5):
    indices =eval(indices)
    #Xs, dfs = standardize_data(X, df)
    # Polynomial özellikleri ekleyin
    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    X_poly_user = poly_features.fit_transform(df)
    
    # Polynomial özelliklerin sütun isimlerini alın
    poly_feature_names = poly_features.get_feature_names_out(input_features=X.columns)

    # Modelin indekslerine göre terimleri belirleyin
    selected_terms = [poly_feature_names[i-1] for i in indices]
    indices = [i - 1 for i in indices]
    
    # Sadece seçilen terimlere sahip yeni bir matris oluşturun
    selected_X_poly = X_poly[:, indices]
    selected_X_poly_user = X_poly_user[:, indices]

    loaded_model = load_model_from_records(model)
    if loaded_model is None:
        print("model yüklenmedi!!!")
    # R^2 skoru için scorer oluştur
    r2_scorer = make_scorer(r2_score)
    # MAE skoru için scorer oluştur
    mae_scorer = make_scorer(mean_absolute_error)

    
    skf = StratifiedKFold(n_splits=cv)

    # Cross-validation ile R^2 ve MAE skorlarını hesapla
    r2_scores = cross_val_score(loaded_model, selected_X_poly, y, cv=cv, scoring=r2_scorer)
    mae_scores = cross_val_score(loaded_model, selected_X_poly, y, cv=cv, scoring=mae_scorer)
    mae = mean_absolute_error(y, loaded_model.predict(selected_X_poly))
    r2 = r2_score(y, loaded_model.predict(selected_X_poly))
    
    # Ortalama skorları al
    mean_r2_score = np.mean(r2_scores)
    mean_mae_score = np.mean(mae_scores)

    # Modelin tahmin yapma yeteneğini değerlendir
    predictions = loaded_model.predict(selected_X_poly)
    predictions[predictions < 0] = 0
    input_pred = loaded_model.predict(selected_X_poly_user)
    input_pred[input_pred < 0] = 0
    return mean_r2_score, mean_mae_score, predictions, input_pred

def main():

    st.title('SAM Analiz Uygulaması')
    models_dir = "models"
    records_dir = "records"

    if not os.listdir(models_dir) or not os.listdir(records_dir):
        st.error("Model eğitmediniz! Lütfen modelleri eğitin ve tekrar deneyin.")
        st.info("5 saniye sonra ana sayfaya yönlendirileceksiniz.")
        time.sleep(5)
        st.experimental_rerun()
    else:

        # Models klasöründeki veritabanlarını listeleme
        modelss = [file for file in os.listdir("models") if file.endswith(".db")]

    # Açılır menü oluşturma
    global model_adi
    model_adi = st.selectbox("Bir Eğitilmiş model Seçiniz:",modelss)
    print(model_adi)
    

    if model_adi != "":  # Kullanıcı bir model seçmişse devam et
        st.header("Model Bilgileri:")
        with st.expander("Detaylar"):
            st.write("----------")
            data  = get_model_info(model_adi)
            st.write("Model Tipi:")
            st.write(data["model_tipi"][0])
            st.write("----------")
            st.write("Bağımlı Değişken: ")
            st.write(data["bagimli_degisken"][0])
            st.write("----------")
            my_list = data["bagimsiz_degiskenler_json"][0]
            result = str(my_list)[1:-1]
            st.write("Bağımsız Değişkenler:")
            st.write(result)
            st.write("----------")
            st.write("Tarih:")
            st.write(data["tarih"][0])
            st.write("----------")
            st.write("Kombinasyon:")
            tuple_deger = data["combo"][0]
            str_deger = str(tuple_deger)
            str_deger = str_deger[3:-4]
            st.write(str_deger)
            st.write("----------")


        selected_option = st.radio("Veri Kaynağını Seçin:", ["CSV Dosyası", "Excel Dosyası", "Veritabanı"])

        if selected_option == "CSV Dosyası":
            uploaded_file = st.file_uploader("Lütfen bir CSV dosyası seçin", type="csv")

            if uploaded_file:
                data = load_data_from_csv(uploaded_file)
                st.write("Veri Seti Önizleme:")
                st.dataframe(data)

                dependent_variable, selected_independent_variables = select_features_and_dependent_variable(data)

                if dependent_variable and selected_independent_variables:
                    y = data[dependent_variable]
                    X = data[selected_independent_variables]
                    df_s = get_user_input(selected_independent_variables)


                    if st.button("Analizi Gerçekleştir"):
                        data1  = get_model_info(model_adi)
                        model_adi1 = data1["model_adi"][0]+".pkl"
                        tuple_deger = data1["combo"][0]
                        str_deger = str(tuple_deger)
                        indices = str_deger[3:-4]
                        st.success(model_adi1)
                        # Modeli değerlendirme
                        if model_tip == "Polynomial":
                            mean_r2, mean_mae, predictions, y_pred_df = evaluate_model_with_cross_validation_for_reg(model_adi1, X, y,df_s,indices)
                        else:
                            mean_r2, mean_mae, predictions, y_pred_df = evaluate_model_with_cross_validation(model_adi1, X, y,df_s)
                        lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)
                        # Modeli değerlendirme
                        mean_r2, mean_mae, predictions, y_pred_df = evaluate_model_with_cross_validation(model_adi1, X, y,df_s)
                        lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)

                        results_df = pd.DataFrame({
                            "Bağımlı Değişken": y.values,
                            "lower_bound": lower_bound,
                            "Tahmin Değerleri": predictions,
                            "upper_bound": upper_bound,
                        })

                        for col in X.columns:
                            results_df[f"{col}"] = X[col].values

                        st.header('Sonuç:')
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.write("Tahmin Değeri:", f"{y_pred_df[0]:.4f}")
                        st.markdown("<hr>", unsafe_allow_html=True)
                        with st.expander("Detaylar"):
                            st.write(f"Ortalama Skor: {mean_r2}")
                            st.markdown("<hr>", unsafe_allow_html=True)

                            st.write(f"Ortalama MAE Skoru: {mean_mae}")
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.dataframe(results_df)
                            reg.plot_prediction_with_ci(y, predictions, lower_bound, upper_bound)






            

        elif selected_option == "Excel Dosyası":
            uploaded_file = st.file_uploader("Lütfen bir Excel dosyası seçin", type=["xls", "xlsx"])

            if uploaded_file:
                data = load_data_from_excel(uploaded_file)
                st.write("Veri Seti Önizleme:")
                st.dataframe(data)

                dependent_variable, selected_independent_variables = select_features_and_dependent_variable(data)

                if dependent_variable and selected_independent_variables:
                    y = data[dependent_variable]
                    X = data[selected_independent_variables]
                    df_s = get_user_input(selected_independent_variables)

                    if st.button("Analizi Gerçekleştir"):
                        data1  = get_model_info(model_adi)
                        model_adi1 = data1["model_adi"][0]+".pkl"
                        tuple_deger = data1["combo"][0]
                        str_deger = str(tuple_deger)
                        indices = str_deger[3:-4]
                        st.success(model_adi1)
                        # Modeli değerlendirme
                        if model_tip == "Polynomial":
                            mean_r2, mean_mae, predictions, y_pred_df = evaluate_model_with_cross_validation_for_reg(model_adi1, X, y,df_s,indices)
                        else:
                            mean_r2, mean_mae, predictions, y_pred_df = evaluate_model_with_cross_validation(model_adi1, X, y,df_s)
                        lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)
                        # Modeli değerlendirme
                        mean_r2, mean_mae, predictions, y_pred_df = evaluate_model_with_cross_validation(model_adi1, X, y,df_s)
                        lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)

                        results_df = pd.DataFrame({
                            "Bağımlı Değişken": y.values,
                            "lower_bound": lower_bound,
                            "Tahmin Değerleri": predictions,
                            "upper_bound": upper_bound,
                        })

                        for col in X.columns:
                            results_df[f"{col}"] = X[col].values

                        st.header('Sonuç:')
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.write("Tahmin Değeri:", f"{y_pred_df[0]:.4f}")
                        st.markdown("<hr>", unsafe_allow_html=True)
                        with st.expander("Detaylar"):
                            st.write(f"Ortalama  Skor: {mean_r2}")
                            st.markdown("<hr>", unsafe_allow_html=True)

                            st.write(f"Ortalama MAE Skoru: {mean_mae}")
                            st.markdown("<hr>", unsafe_allow_html=True)
                            st.dataframe(results_df)
                            reg.plot_prediction_with_ci(y, predictions, lower_bound, upper_bound)






        elif selected_option == "Veritabanı":
            database_files = [f for f in os.listdir("databases/") if f.endswith(".db")]
            selected_database = st.selectbox("Veritabanını Seçin:", database_files)

            conn = sqlite3.connect(f"databases/{selected_database}")
            cursor = conn.cursor()

            tables = [table[0] for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")]
            selected_table = st.selectbox("Veritabanındaki Tabloyu Seçin:", tables)

            data = load_data_from_database(f"databases/{selected_database}", selected_table)

            st.write("Veri Seti Önizleme:")
            st.dataframe(data)

            dependent_variable, selected_independent_variables = select_features_and_dependent_variable(data)

            if dependent_variable and selected_independent_variables:
                y = data[dependent_variable]
                X = data[selected_independent_variables]
                df_s = get_user_input(selected_independent_variables)
                
                max_degree = None


                if st.button("Analizi Gerçekleştir"):
                    data1  = get_model_info(model_adi)
                    model_adi1 = data1["model_adi"][0]+".pkl"
                    model_tip = data1["model_tipi"][0]

                    tuple_deger = data1["combo"][0]
                    str_deger = str(tuple_deger)
                    indices = str_deger[3:-4]
                    st.success(model_adi1)
                    # Modeli değerlendirme
                    if model_tip == "Polynomial":
                        mean_r2, mean_mae, predictions, y_pred_df = evaluate_model_with_cross_validation_for_reg(model_adi1, X, y,df_s,indices)
                    else:
                        mean_r2, mean_mae, predictions, y_pred_df = evaluate_model_with_cross_validation(model_adi1, X, y,df_s)
                    lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)

                    results_df = pd.DataFrame({
                        "Bağımlı Değişken": y.values,
                        "lower_bound": lower_bound,
                        "Tahmin Değerleri": predictions,
                        "upper_bound": upper_bound,
                    })

                    for col in X.columns:
                        results_df[f"{col}"] = X[col].values

                    st.header('Sonuç:')
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.write("Tahmin Değeri:", f"{y_pred_df[0]:.4f}")
                    st.markdown("<hr>", unsafe_allow_html=True)
                    with st.expander("Detaylar"):

                        st.write(f"Ortalama  Skor: {mean_r2}")
                        st.markdown("<hr>", unsafe_allow_html=True)

                        st.write(f"Ortalama MAE Skoru: {mean_mae}")
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.dataframe(results_df)
                        reg.plot_prediction_with_ci(y, predictions, lower_bound, upper_bound)
            conn.close()
if __name__ == "__main__":
    main()
