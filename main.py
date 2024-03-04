import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
import os
import sqlite3
import regresyon as reg  # Import your regresyon module here
import ml2 as ml  # Import your machine learning model here
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle
import json
import datetime
global modeller1
modeller1 = {}


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

def standardize_data(X_train, X_val):
    scaler = StandardScaler()
    standardized_X_train = scaler.fit_transform(X_train)
    standardized_X_val = scaler.fit_transform(X_val)
    return pd.DataFrame(standardized_X_train, columns=X_train.columns),pd.DataFrame(standardized_X_val, columns=X_train.columns)

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

# Add SVR function call here
def perform_svr_analysis(X, y ):
    st.write("Performing SVR Analysis")
    # Add your SVR function call here
    # Veriyi train ve validation setlerine bölme
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_trains, X_vals  = standardize_data(X_train, X_val)

    # Parametre aralıkları
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 0.2],
        'kernel': ['linear', 'rbf', 'poly']
    }

    # SVR modelini eğitme
    model, r2_train = ml.train_svr(X_trains, y_train, param_grid)

    # SVR modelini değerlendirme
    r2_val, mae_val, y_pred_val = ml.evaluate_svr(model, X_vals, y_val)
    #svr, r2, mae, y_pred = ml.evaluate_svr(X, y, cv=5)
    lower_bound, upper_bound = reg.calculate_bootstrap_ci(X_vals, y_val)
    model_tipi = "SVR"
    r2_degeri = f"{r2_val:.4f}"
    modeller1[model_tipi] = r2_degeri

    results_df = pd.DataFrame({
            "Bağımlı Değişken": y_val.values,
            "lower_bound": lower_bound,
            "Tahmin Değerleri": y_pred_val,
            "upper_bound": upper_bound,
        })

    for col in X.columns:
        results_df[f"{col}"] = X_val[col].values


    with st.expander("Detaylar"):
        st.write("R$^2$:", f"{r2_val:.4f}")
        st.markdown("<hr>", unsafe_allow_html=True)

        st.write("Mean Absolute Error:",f"{mae_val:.4f}" )
        st.markdown("<hr>", unsafe_allow_html=True)
        st.dataframe(results_df)
        reg.plot_prediction_with_ci(y_val, y_pred_val, lower_bound, upper_bound)
        


    return model
# Add Random Forest function call here
def perform_random_forest_analysis(X, y):
    st.write("Performing Random Forest Analysis")
    # Add your Random Forest function call here
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_trains, X_vals = standardize_data(X_train, X_val)
    
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model, r2_train_rf = ml.train_random_forest(X_trains, y_train, param_grid_rf)
    r2_val, mae_val, y_pred_val = ml.evaluate_svr(model, X_vals, y_val)
    lower_bound, upper_bound = reg.calculate_bootstrap_ci(X_vals, y_val)
    model_tipi = "Random Forest"
    r2_degeri = f"{r2_val:.4f}"
    modeller1[model_tipi] = r2_degeri

    results_df = pd.DataFrame({
            "Bağımlı Değişken": y_val.values,
            "lower_bound": lower_bound,
            "Tahmin Değerleri": y_pred_val,
            "upper_bound": upper_bound,
        })

    for col in X_val.columns:
        results_df[f"{col}"] = X_val[col].values


    with st.expander("Detaylar"):
        st.write("R$^2$:", f"{r2_val:.4f}")
        st.markdown("<hr>", unsafe_allow_html=True)

        st.write("Mean Absolute Error:",f"{mae_val:.4f}" )
        st.markdown("<hr>", unsafe_allow_html=True)

        st.dataframe(results_df)
        reg.plot_prediction_with_ci(y_val, y_pred_val, lower_bound, upper_bound)

    return model   

# Add Decision Tree function call here
def perform_decision_tree_analysis(X, y):
    st.write("Performing Decision Tree Analysis")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_trains, X_vals = standardize_data(X_train, X_val)
    param_grid_dt = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
    # dt modelini eğitme
    model, r2_train = ml.train_decision_tree(X_trains, y_train, param_grid_dt)

    # dt modelini değerlendirme
    r2_val, mae_val, y_pred_val = ml.evaluate_svr(model, X_vals, y_val)
    lower_bound, upper_bound = reg.calculate_bootstrap_ci(X_vals, y_val)
    model_tipi = "Decision Tree"
    r2_degeri = f"{r2_val:.4f}"
    modeller1[model_tipi] = r2_degeri

    results_df = pd.DataFrame({
            "Bağımlı Değişken": y_val.values,
            "lower_bound": lower_bound,
            "Tahmin Değerleri": y_pred_val,
            "upper_bound": upper_bound,
        })

    for col in X.columns:
        results_df[f"{col}"] = X_val[col].values

    

    with st.expander("Detaylar"):
        st.write("R$^2$:", f"{r2_val:.4f}")
        st.markdown("<hr>", unsafe_allow_html=True)

        st.write("Mean Absolute Error:",f"{mae_val:.4f}" )
        st.markdown("<hr>", unsafe_allow_html=True)

        st.dataframe(results_df)
        reg.plot_prediction_with_ci(y_val, y_pred_val, lower_bound, upper_bound)

    return model

def perform_regression_analysis(X, y, max_degree):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    X_trains, X_vals = standardize_data(X_train, X_val)
    start_time = time.time()
    top_10_combinations = reg.find_best_polynomial_combinations(X_trains, y_train, max_degree, top_n=1, max_terms=20)

    if top_10_combinations:
        global combo
        combo, _ = top_10_combinations[0]
 
        reg.klasor_islemleri(str(combo))
        termss, r2, mae, y_pred,coeff,intcept,model = reg.generate_polynomial_model(X_train, y_train, combo, max_degree)
        math_model = reg.generate_math_model(termss, coeff, intcept)
        lower_bound, upper_bound = reg.calculate_bootstrap_ci(X_val, y_val)
        y_pred_val = reg.make_prediction(X_trains, y_train, combo, max_degree, X_vals)
        model_tipi = "polinomiyal Regresyon"
        r2_degeri = f"{r2:.4f}"
        modeller1[model_tipi] = r2_degeri

    else:
        st.warning("Lütfen analiz yapmak için geçerli bir dosya seçin.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Elapsed Time: {elapsed_time} seconds")
    return model

def choice_the_model(regression_type,X, y, max_degree):
    # Choose the appropriate regression type based on the user's selection
    if regression_type == "SVR":
        model = perform_svr_analysis(X, y)
    elif regression_type == "Random Forest":
        model = perform_random_forest_analysis(X, y)
    elif regression_type == "Decision Tree":
        model = perform_decision_tree_analysis(X, y)
    elif regression_type == "Polynomial":
       model = perform_regression_analysis(X, y, max_degree)

    return model

def perform_analysis_and_update_session_state(regression_type, X, y, max_degree):
    # Access session state or initialize it if not exists
    session_state = st.session_state
    session_state.modell = choice_the_model(regression_type, X, y, max_degree)




def kaydet_ve_ekle(model, model_adi, bagimli_degisken, bagimsiz_degiskenler, model_tipi,combo):
    """
    Eğitilmiş regresyon modelini kaydeder ve veritabanına ekler.

    Parametreler:
        model: Eğitilmiş regresyon modeli.
        model_adi: Modelin adı.
        bagimli_degisken: Bağımlı değişkenin adı.
        bagimsiz_degiskenler: Bağımsız değişkenlerin bir listesi.
        model_tipi: Regresyon modelinin tipi.
    """

    # Kayıt dosyasına kaydetme
    if not os.path.exists("records"):
        # Klasör yoksa oluşturma
        os.makedirs("records")
    with open(f"records/{model_adi}.pkl", "wb") as f:
        pickle.dump(model, f)

    # Bağımsız değişkenleri JSON'a dönüştürme
    json_str = json.dumps(bagimsiz_degiskenler)

    # Veritabanı bağlantısı
    if not os.path.exists("models"):
        # Klasör yoksa oluşturma
        os.makedirs("models")
    con = sqlite3.connect(f"models/{model_adi}.db")
    cursor = con.cursor()

    # Veritabanı oluşturma
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_info (
            model_adi TEXT NOT NULL,
            bagimli_degisken TEXT NOT NULL,
            bagimsiz_degiskenler_json TEXT NOT NULL,
            model_tipi TEXT NOT NULL,
            tarih TEXT NOT NULL,
            dosya_adi TEXT NOT NULL,
            combo TEXT NOT NULL
        )
    """)

    # Veritabanına ekleme
    cursor.execute("""
        INSERT INTO model_info (model_adi, bagimli_degisken, bagimsiz_degiskenler_json, model_tipi, tarih, dosya_adi, combo)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (model_adi, bagimli_degisken, json_str, model_tipi, datetime.datetime.now().strftime("%Y-%m-%d"), f"{model_adi}.pkl",combo))
    con.commit()
    con.close()

    # Mesaj
    st.success(f"Model başarıyla kaydedildi: {model_adi}.pkl")
    st.success(f"Veritabanına başarıyla eklendi: {model_adi}")



def main():

    st.title('SAM Analiz Uygulaması')
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

            if st.button("Modeli Eğit"):
                modeller = ["Polynomial", "SVR", "Random Forest", "Decision Tree"]
                

                for model_tipi in modeller:

                    max_degree = None
                    if model_tipi == "Polynomial":
                        max_degree = 3

                    # Klasörün varlığını kontrol etme
                    if not os.path.exists("r2"):
                        # Klasör yoksa oluşturma
                        os.makedirs("r2")

                    # Klasördeki dosyaları silme
                    for filename in os.listdir("r2"):
                        # Dosya yolunu oluşturma
                        file_path = os.path.join("r2", filename)
                        # Dosyayı silme
                        os.remove(file_path)

                    perform_analysis_and_update_session_state(model_tipi, X, y, max_degree)

                    modell = st.session_state.modell
                    # Check if modell is not None before accessing its attributes
                    if modell is not None:
                        tarih = datetime.datetime.now().strftime("%Y-%m-%d")
                        model_adi = model_tipi+str(selected_independent_variables) + "_"+ tarih
                        if model_tipi == "Polynomial":
                            combo = reg.degeri_oku_ve_yazdir()
                        else:
                            combo = "1"
                        kaydet_ve_ekle(modell, model_adi, dependent_variable, selected_independent_variables, model_tipi,str(combo))

                    else:
                        st.warning("Modell is not defined. Run analysis first.")

                for model_tipi, r2_degeri in modeller1.items():
                    st.write(f"{model_tipi}: {r2_degeri}")
                en_yuksek_r2 = 0

                # En yüksek r2 değerine sahip model tipini bul
                en_iyi_model_tipi = None
                for model_tipi, r2_degeri in modeller1.items():
                    if float(r2_degeri) > float(en_yuksek_r2):
                        en_yuksek_r2 = r2_degeri
                        en_iyi_model_tipi = model_tipi
                st.success(f"Mevcut modellerimiz arasında {en_iyi_model_tipi}: {en_yuksek_r2} ile en iyi performansı göstermektedir.")


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

                if st.button("Modeli Eğit"):
                    modeller = ["Polynomial", "SVR", "Random Forest", "Decision Tree"]
                    

                    for model_tipi in modeller:

                        max_degree = None
                        if model_tipi == "Polynomial":
                            max_degree = 3

                        # Klasörün varlığını kontrol etme
                        if not os.path.exists("r2"):
                            # Klasör yoksa oluşturma
                            os.makedirs("r2")

                        # Klasördeki dosyaları silme
                        for filename in os.listdir("r2"):
                            # Dosya yolunu oluşturma
                            file_path = os.path.join("r2", filename)
                            # Dosyayı silme
                            os.remove(file_path)

                        perform_analysis_and_update_session_state(model_tipi, X, y, max_degree)

                        modell = st.session_state.modell
                        # Check if modell is not None before accessing its attributes
                        if modell is not None:
                            tarih = datetime.datetime.now().strftime("%Y-%m-%d")
                            model_adi = model_tipi+str(selected_independent_variables) + "_"+ tarih
                            if model_tipi == "Polynomial":
                                combo = reg.degeri_oku_ve_yazdir()
                            else:
                                combo = "1"
                            kaydet_ve_ekle(modell, model_adi, dependent_variable, selected_independent_variables, model_tipi,str(combo))

                        else:
                            st.warning("Modell is not defined. Run analysis first.")

                

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




            if st.button("Modeli Eğit"):
                modeller = ["Polynomial", "SVR", "Random Forest", "Decision Tree"]
                

                for model_tipi in modeller:

                    max_degree = None
                    if model_tipi == "Polynomial":
                        max_degree = 3

                    # Klasörün varlığını kontrol etme
                    if not os.path.exists("r2"):
                        # Klasör yoksa oluşturma
                        os.makedirs("r2")

                    # Klasördeki dosyaları silme
                    for filename in os.listdir("r2"):
                        # Dosya yolunu oluşturma
                        file_path = os.path.join("r2", filename)
                        # Dosyayı silme
                        os.remove(file_path)

                    perform_analysis_and_update_session_state(model_tipi, X, y, max_degree)

                    modell = st.session_state.modell
                    # Check if modell is not None before accessing its attributes
                    if modell is not None:
                        tarih = datetime.datetime.now().strftime("%Y-%m-%d")
                        model_adi = model_tipi+str(selected_independent_variables) + "_"+ tarih
                        if model_tipi == "Polynomial":
                            combo = reg.degeri_oku_ve_yazdir()
                        else:
                            combo = "1"
                        kaydet_ve_ekle(modell, model_adi, dependent_variable, selected_independent_variables, model_tipi,str(combo))

                    else:
                        st.warning("Modell is not defined. Run analysis first.")
                    

        conn.close()
if __name__ == "__main__":
    main()
