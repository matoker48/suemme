import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st
import os
import sqlite3
import regresyon as reg
def plot_prediction_with_ci(y, y_pred, lower_bound, upper_bound):
    # Create a DataFrame containing real values, predictions, and confidence intervals
    results_df = pd.DataFrame({
        "Real Values": y.values,
        "Predicted Values": y_pred,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound
    })

    # Reset the index
    results_df.reset_index(drop=True, inplace=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use np.arange for one-dimensional x-axis values
    x_values = np.arange(len(results_df))
    
    ax.plot(x_values, results_df["Real Values"], label="Real Values", marker='o')
    ax.plot(x_values, results_df["Predicted Values"], label="Predicted Values", marker='o')
    ax.fill_between(x_values, results_df["Lower Bound"], results_df["Upper Bound"], color='gray', alpha=0.3, label="Confidence Interval")
    
    ax.set_xlabel("Observation Indices")
    ax.set_ylabel("Values")
    ax.set_title("Predicted Values and Confidence Interval")
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)
combo = None
max_degree = None
X = None 
y = None

# Sayfa başlığı
st.title('Regresyon Analizi Uygulaması')

# Dosya veya veritabanı türünü seçme
selected_option = st.radio("Veri Kaynağını Seçin:", ["CSV Dosyası", "Excel Dosyası", "Veritabanı"])

if selected_option == "CSV Dosyası":
    uploaded_file = st.file_uploader("Lütfen bir CSV dosyası seçin", type="csv")

    if uploaded_file is not None:
        # Ayırıcıyı belirt
        data = pd.read_csv(uploaded_file, sep=';')

        st.write("Veri Seti Önizleme:")
        st.dataframe(data.head())

        # Kullanıcıdan bağımlı değişken seçimini al
        dependent_variable = st.selectbox("Bağımlı Değişken:", data.columns)

        # Bağımlı değişkeni seçilen değişken olarak ayarla
        independent_variables = [col for col in data.columns if col != dependent_variable]

        # Kullanıcının seçtiği bağımlı değişkenin yanındaki checkbox'lar
        # ile bağımsız değişkenleri seçmesini sağla
        selected_independent_variables = st.multiselect("Bağımsız Değişkenleri Seçin:", independent_variables)

        if not dependent_variable or not selected_independent_variables:
            st.warning("Lütfen bağımlı ve bağımsız değişkenleri seçin.")
        else:
            y = data[dependent_variable]
            X = data[selected_independent_variables]
            scaler = StandardScaler()
            user_input = {}
            for col in selected_independent_variables:
                value = st.number_input(f"{col} Değerini Girin:")
                user_input[col] = [value]
            df = pd.DataFrame(user_input)
            
            # Standartlaştır
            scaler =StandardScaler()
            scaler.fit(X)
            standardized_X = scaler.transform(X)
            standardized_df = scaler.transform(df)
            # DataFrame'e çevir
            X = pd.DataFrame(standardized_X, columns=X.columns)
            df_s = pd.DataFrame(standardized_df, columns=df.columns)
            st.dataframe(df_s)
            # Tahmin değerlerini içeren DataFrame'i oluşturun


            # Polynomial derecesi seçimi
            max_degree = st.slider("Polinom Derecesi Seç", min_value=1, max_value=5, value=2)
            # Measure the start time
            start_time = time.time()
            top_10_combinations = reg.find_best_polynomial_combinations(X, y,max_degree , top_n=1, max_terms=20)
            # Measure the end time
            end_time = time.time()

            if st.button("Analizi Gerçekleştir"):
                # Polynomial Regression Analysis
                st.header("Cıktı:")




                # Calculate the elapsed time
                elapsed_time = end_time - start_time

                # Print the elapsed time
                st.write(f"Elapsed Time: {elapsed_time} seconds")
                com=[]
                if top_10_combinations:
                    combo, _ = top_10_combinations[0]  # Assuming top_10_combinations now contains only one combination
                    # Generate Polynomial Model
                    _, _, _, y_pred = reg.generate_polynomial_model(X, y, combo, max_degree)
                    lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)

                    results_df = pd.DataFrame({
                        "Bağımlı Değişken": y.values,
                        "lower_bound":lower_bound,
                        "Tahmin Değerleri": y_pred,
                        "upper_bound":upper_bound,
                    })
                    
                    # Bağımsız değişkenleri dinamik olarak DataFrame'e ekleyin
                    for col in data[selected_independent_variables].columns:
                        results_df[f"{col}"] = data[selected_independent_variables][col].values
                    
                    st.dataframe(results_df)
                    # Tahmin yapın
                    prediction = reg.make_prediction(X, y, combo, max_degree, df_s)
                    # Fonksiyonunuzu çağırın ve grafiği çizin
                    reg.plot_prediction_with_ci(y, y_pred, lower_bound, upper_bound)

                    # Tahmin sonuçlarını gösterin
                    st.write("Tahmin Değeri:", prediction)

                else:
                    st.warning("Lütfen analiz yapmak için geçerli bir dosya seçin.")

elif selected_option == "Excel Dosyası":
    uploaded_file = st.file_uploader("Lütfen bir Excel dosyası seçin", type=["xls", "xlsx"])

    dependent_variable = None
    selected_independent_variables = None

    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)

        st.write("Veri Seti Önizleme:")
        st.dataframe(data.head())

        dependent_variable = st.selectbox("Bağımlı Değişken:", data.columns)

        independent_variables = [col for col in data.columns if col != dependent_variable]

        selected_independent_variables = st.multiselect("Bağımsız Değişkenleri Seçin:", independent_variables)

    if not dependent_variable or not selected_independent_variables:
        st.warning("Lütfen bağımlı ve bağımsız değişkenleri seçin.")
   
    else:
        y = data[dependent_variable]
        X = data[selected_independent_variables]
        scaler = StandardScaler()
        user_input = {}
        for col in selected_independent_variables:
            value = st.number_input(f"{col} Değerini Girin:")
            user_input[col] = [value]
        df = pd.DataFrame(user_input)
        
        # Standartlaştır
        scaler =StandardScaler()
        scaler.fit(X)
        standardized_X = scaler.transform(X)
        standardized_df = scaler.transform(df)
        # DataFrame'e çevir
        X = pd.DataFrame(standardized_X, columns=X.columns)
        df_s = pd.DataFrame(standardized_df, columns=df.columns)
        st.dataframe(df_s)
        # Tahmin değerlerini içeren DataFrame'i oluşturun


        # Polynomial derecesi seçimi
        max_degree = st.slider("Polinom Derecesi Seç", min_value=1, max_value=5, value=2)
        # Measure the start time
        start_time = time.time()
        top_10_combinations = reg.find_best_polynomial_combinations(X, y,max_degree , top_n=1, max_terms=20)
        # Measure the end time
        end_time = time.time()

        if st.button("Analizi Gerçekleştir"):
            # Polynomial Regression Analysis
            st.header("Çıktı")




            # Calculate the elapsed time
            elapsed_time = end_time - start_time

            # Print the elapsed time
            st.write(f"Elapsed Time: {elapsed_time} seconds")
            com=[]
            if top_10_combinations:
                combo, _ = top_10_combinations[0]  # Assuming top_10_combinations now contains only one combination
                # Generate Polynomial Model
                _, _, _, y_pred = reg.generate_polynomial_model(X, y, combo, max_degree)
                lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)


                
                results_df = pd.DataFrame({
                    "Bağımlı Değişken": y.values,
                    "lower_bound":lower_bound,
                    "Tahmin Değerleri": y_pred,
                    "upper_bound":upper_bound,
                })
                
                # Bağımsız değişkenleri dinamik olarak DataFrame'e ekleyin
                for col in data[selected_independent_variables].columns:
                    results_df[f"{col}"] = data[selected_independent_variables][col].values
                
                st.dataframe(results_df)
                # Tahmin yapın
                prediction = reg.make_prediction(X, y, combo, max_degree, df_s)
                # Fonksiyonunuzu çağırın ve grafiği çizin
                plot_prediction_with_ci(y, y_pred, lower_bound, upper_bound)

                # Tahmin sonuçlarını gösterin
                st.write("Tahmin Değeri:", prediction)

            else:
                st.warning("Lütfen analiz yapmak için geçerli bir dosya seçin.")
elif selected_option == "Veritabanı":
    # Veritabanı klasöründeki dosyaları listeleme
    database_files = [f for f in os.listdir("databases/") if f.endswith(".db")]
    selected_database = st.selectbox("Veritabanını Seçin:", database_files)

    # Veritabanına bağlantı
    conn = sqlite3.connect(f"databases/{selected_database}")
    cursor = conn.cursor()

    # Veritabanındaki tabloları listeleme
    tables = [table[0] for table in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")]
    selected_table = st.selectbox("Veritabanındaki Tabloyu Seçin:", tables)

    # Veriyi DataFrame'e alma
    data = pd.read_sql_query(f"SELECT * FROM {selected_table};", conn)

    # Veri Seti Önizleme
    st.write("Veri Seti Önizleme:")
    st.dataframe(data.head())

    dependent_variable = st.selectbox("Bağımlı Değişken:", data.columns)

    independent_variables = [col for col in data.columns if col != dependent_variable]

    # Kullanıcının girdiği bağımsız değişken değerlerini alıp tahmin yapma

    selected_independent_variables = st.multiselect("Bağımsız Değişkenleri Seçin:", independent_variables)


    if not dependent_variable or not selected_independent_variables:
        st.warning("Lütfen bağımlı ve bağımsız değişkenleri seçin.")
    else:
        y = data[dependent_variable]
        X = data[selected_independent_variables]
        scaler = StandardScaler()
        user_input = {}
        for col in selected_independent_variables:
            value = st.number_input(f"{col} Değerini Girin:")
            user_input[col] = [value]
        df = pd.DataFrame(user_input)
        
        # Standartlaştır
        scaler =StandardScaler()
        scaler.fit(X)
        standardized_X = scaler.transform(X)
        standardized_df = scaler.transform(df)
        # DataFrame'e çevir
        X = pd.DataFrame(standardized_X, columns=X.columns)
        df_s = pd.DataFrame(standardized_df, columns=df.columns)
        st.dataframe(df_s)
        # Tahmin değerlerini içeren DataFrame'i oluşturun


        # Polynomial derecesi seçimi
        max_degree = st.slider("Polinom Derecesi Seç", min_value=1, max_value=5, value=2)
        # Measure the start time
        start_time = time.time()
        top_10_combinations = reg.find_best_polynomial_combinations(X, y,max_degree , top_n=1, max_terms=20)
        # Measure the end time
        end_time = time.time()

        if st.button("Analizi Gerçekleştir"):
            # Polynomial Regression Analysis
            st.header("Çıktı")




            # Calculate the elapsed time
            elapsed_time = end_time - start_time

            # Print the elapsed time
            st.write(f"Elapsed Time: {elapsed_time} seconds")
            com = []

            if top_10_combinations:
                combo, _ = top_10_combinations[0]  # Assuming top_10_combinations now contains only one combination
                # Generate Polynomial Model
                _, _, _, y_pred = reg.generate_polynomial_model(X, y, combo, max_degree)
                lower_bound, upper_bound = reg.calculate_bootstrap_ci(X, y)
               
                results_df = pd.DataFrame({
                    "Bağımlı Değişken": y.values,
                    "lower_bound":lower_bound,
                    "Tahmin Değerleri": y_pred,
                    "upper_bound":upper_bound,
                })
                
                # Bağımsız değişkenleri dinamik olarak DataFrame'e ekleyin
                for col in data[selected_independent_variables].columns:
                    results_df[f"{col}"] = data[selected_independent_variables][col].values
                
                st.dataframe(results_df)
                # Tahmin yapın
                prediction = reg.make_prediction(X, y, combo, max_degree, df_s)
                # Fonksiyonunuzu çağırın ve grafiği çizin
                plot_prediction_with_ci(y, y_pred, lower_bound, upper_bound)

                # Tahmin sonuçlarını gösterin
                st.write("Tahmin Değeri:", prediction)



               

            else:
                st.warning("Lütfen analiz yapmak için geçerli bir dosya seçin.")


    # Veritabanı bağlantısını kapatma
    conn.close()



