import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
import os
import sqlite3
import matplotlib.pyplot as plt
import regresyon as reg  # Import your regresyon module here
import ml  # Import your machine learning model here

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

def standardize_data(X, df_s):
    scaler = StandardScaler()
    standardized_X = scaler.fit_transform(X)
    standardized_df_s = scaler.transform(df_s)
    return pd.DataFrame(standardized_X, columns=X.columns), pd.DataFrame(standardized_df_s, columns=df_s.columns), scaler

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
def perform_svr_analysis(X, y, df_s, scaler):
    st.write("Performing SVR Analysis")
    # Add your SVR function call here
    svr, r2, mae, y_pred = ml.evaluate_svr(X, y, cv=5)
    lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)

    results_df = pd.DataFrame({
            "Bağımlı Değişken": y.values,
            "lower_bound": lower_bound,
            "Tahmin Değerleri": y_pred,
            "upper_bound": upper_bound,
        })

    for col in X.columns:
        results_df[f"{col}"] = X[col].values

    

    prediction = ml.predict_with(svr,df_s)
    st.header('Sonuç:')
    st.markdown("<hr>", unsafe_allow_html=True)

    st.write("Tahmin Değeri:", prediction)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.dataframe(results_df)
    reg.plot_prediction_with_ci(y, y_pred, lower_bound, upper_bound)



# Add Random Forest function call here
def perform_random_forest_analysis(X, y, df_s, scaler):
    st.write("Performing Random Forest Analysis")
    # Add your Random Forest function call here
    # Add your SVR function call here
    rf, r2, mae, y_pred = ml.evaluate_random_forest(X, y, cv=5)
    lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)

    results_df = pd.DataFrame({
            "Bağımlı Değişken": y.values,
            "lower_bound": lower_bound,
            "Tahmin Değerleri": y_pred,
            "upper_bound": upper_bound,
        })

    for col in X.columns:
        results_df[f"{col}"] = X[col].values


    prediction = ml.predict_with(rf,df_s)
    st.header('Sonuç:')
    st.markdown("<hr>", unsafe_allow_html=True)

    st.write("Tahmin Değeri:", prediction)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.dataframe(results_df)
    reg.plot_prediction_with_ci(y, y_pred, lower_bound, upper_bound)

    

# Add Decision Tree function call here
def perform_decision_tree_analysis(X, y, df_s, scaler):
    st.write("Performing Decision Tree Analysis")
    # Add your Decision Tree function call here
    # Add your SVR function call here
    dt, r2, mae, y_pred = ml.evaluate_decision_tree(X, y, cv=5)
    lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)

    results_df = pd.DataFrame({
            "Bağımlı Değişken": y.values,
            "lower_bound": lower_bound,
            "Tahmin Değerleri": y_pred,
            "upper_bound": upper_bound,
        })

    for col in X.columns:
        results_df[f"{col}"] = X[col].values

    

    prediction = ml.predict_with(dt,df_s)
    st.header('Sonuç:')
    st.markdown("<hr>", unsafe_allow_html=True)

    st.write("Tahmin Değeri:", prediction)
    st.markdown("<hr>", unsafe_allow_html=True)

    st.dataframe(results_df)
    reg.plot_prediction_with_ci(y, y_pred, lower_bound, upper_bound)

    

def perform_regression_analysis(X, y, max_degree, df_s, scaler):

    start_time = time.time()
    top_10_combinations = reg.find_best_polynomial_combinations(X, y, max_degree, top_n=1, max_terms=20)

    if top_10_combinations:
        combo, _ = top_10_combinations[0]
        _, _, _, y_pred = reg.generate_polynomial_model(X, y, combo, max_degree)
        lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)

        results_df = pd.DataFrame({
            "Bağımlı Değişken": y.values,
            "lower_bound": lower_bound,
            "Tahmin Değerleri": y_pred,
            "upper_bound": upper_bound,
        })

        for col in X.columns:
            results_df[f"{col}"] = X[col].values

        
        
        prediction = reg.make_prediction(X, y, combo, max_degree, df_s)
        st.header('Sonuç:')

        st.markdown("<hr>", unsafe_allow_html=True)

        st.write("Tahmin Değeri:", prediction[0])
        st.markdown("<hr>", unsafe_allow_html=True)

        st.dataframe(results_df)
        reg.plot_prediction_with_ci(y, y_pred, lower_bound, upper_bound)


    else:
        st.warning("Lütfen analiz yapmak için geçerli bir dosya seçin.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Elapsed Time: {elapsed_time} seconds")

def choice_the_model(regression_type,X, y, max_degree, df_s, scaler):
    # Choose the appropriate regression type based on the user's selection
    if regression_type == "SVR":
        perform_svr_analysis(X, y, df_s, scaler)
    elif regression_type == "Random Forest":
        perform_random_forest_analysis(X, y, df_s, scaler)
    elif regression_type == "Decision Tree":
        perform_decision_tree_analysis(X, y, df_s, scaler)
    elif regression_type == "Polynomial":
        perform_regression_analysis(X, y, max_degree, df_s, scaler)

def main():
    st.title('SAM Analiz Uygulaması')
    # Add a selection box for regression type
    regression_type = st.selectbox("Bir Model Seçiniz:", ["Polynomial", "SVR", "Random Forest", "Decision Tree"])

    selected_option = st.radio("Veri Kaynağını Seçin:", ["CSV Dosyası", "Excel Dosyası", "Veritabanı"])

    if selected_option == "CSV Dosyası":
        uploaded_file = st.file_uploader("Lütfen bir CSV dosyası seçin", type="csv")

        if uploaded_file:
            data = load_data_from_csv(uploaded_file)
            st.write("Veri Seti Önizleme:")
            st.dataframe(data.head())

            dependent_variable, selected_independent_variables = select_features_and_dependent_variable(data)

            if dependent_variable and selected_independent_variables:
                y = data[dependent_variable]
                X = data[selected_independent_variables]
                df_s = get_user_input(selected_independent_variables)

                X, df_s, scaler = standardize_data(X, df_s)
                max_degree = None
                if regression_type == "Polynomial":
                    max_degree = st.slider("Polinom Derecesi Seç", min_value=1, max_value=3, value=2)
                if st.button("Analizi Gerçekleştir"):
                    choice_the_model(regression_type,X, y, max_degree, df_s, scaler)

    elif selected_option == "Excel Dosyası":
        uploaded_file = st.file_uploader("Lütfen bir Excel dosyası seçin", type=["xls", "xlsx"])

        if uploaded_file:
            data = load_data_from_excel(uploaded_file)
            st.write("Veri Seti Önizleme:")
            st.dataframe(data.head())

            dependent_variable, selected_independent_variables = select_features_and_dependent_variable(data)

            if dependent_variable and selected_independent_variables:
                y = data[dependent_variable]
                X = data[selected_independent_variables]
                df_s = get_user_input(selected_independent_variables)

                X, df_s, scaler = standardize_data(X, df_s)
                max_degree = None
                if regression_type == "Polynomial":
                    max_degree = st.slider("Polinom Derecesi Seç", min_value=1, max_value=3, value=2)

                if st.button("Analizi Gerçekleştir"):
                    choice_the_model(regression_type,X, y, max_degree, df_s, scaler)

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

            X, df_s, scaler = standardize_data(X, df_s)
            max_degree = None
            if regression_type == "Polynomial":
                max_degree = st.slider("Polinom Derecesi Seç", min_value=1, max_value=3, value=2)


            if st.button("Analizi Gerçekleştir"):
                choice_the_model(regression_type,X, y, max_degree, df_s, scaler)

        conn.close()

if __name__ == "__main__":
    main()
