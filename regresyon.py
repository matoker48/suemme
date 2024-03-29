import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from itertools import combinations
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import streamlit as st
import os
import sqlite3

#---------------------------------------------------------------------------
def calculate_r_squared(y, y_pred):
    ss_total = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - y_pred) ** 2).sum()
    return 1 - (ss_res / ss_total)

def calculate_k(observation_count):
    if observation_count < 200:
        result = 5
    else:
        result = observation_count // 40
    
    return int(result)

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
def find_best_polynomial_combinations(X, y, max_degree, top_n, max_terms):
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
            if terms:
                if all(p <= 0.05 for p in p_values) and all(p <=5  for p in vif_data["VIF"]):
                    all_combinations.append((combo, r2))

    all_combinations.sort(key=lambda x: x[1], reverse=True)
    top_combinations = all_combinations[:top_n]

    return top_combinations




def generate_polynomial_model(X, y, indices, degree):
    # Polynomial özellikleri ekleyin
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    # Polynomial özelliklerin sütun isimlerini alın
    poly_feature_names = poly_features.get_feature_names_out(input_features=X.columns)

    # Modelin indekslerine göre terimleri belirleyin
    selected_terms = [poly_feature_names[i-1] for i in indices]
    
    indices = [i - 1 for i in indices]

    # Sadece seçilen terimlere sahip yeni bir matris oluşturun
    selected_X_poly = X_poly[:, indices]
    selected_X_poly_df = pd.DataFrame(selected_X_poly, columns=selected_terms)

    # Modeli oluşturun
    model = LinearRegression()
    cv = calculate_k(len(X))
    model.fit(selected_X_poly, y)
    # Cross-validation ile R2 skorunu hesapla
    r2_scores = cross_val_score(model, selected_X_poly, y, scoring='r2', cv=KFold(n_splits=cv, shuffle=True, random_state=42))
    r2 = np.mean(r2_scores)
    print(f"r2 score: {r2}")

    # Cross-validation ile MAE hesapla
    mae_scores = cross_val_score(model, selected_X_poly, y, scoring='neg_mean_absolute_error', cv=KFold(n_splits=cv, shuffle=True, random_state=42))
    mae = np.mean(-mae_scores)
    print(f"Validation setinde MAE: {mae}")

    coeff = model.coef_
    intcept = model.intercept_


    # Tahminleri alın
    y_pred = cross_val_predict(model, selected_X_poly, y, cv=cv)

    return selected_terms, r2, mae, y_pred,coeff,intcept,model





def generate_math_model(terms, coefficients, intercept):
  """
  Verilen terimleri, katsayıları ve kesme noktası değerini kullanarak bir regresyon modeli oluşturur.

  Args:
      terms (list): Terimlerin bir listesini içeren Python listesi.
      coefficients (list): Katsayıların bir listesini içeren Python listesi.
      intercept (float): Kesme noktası değeri.

  Returns:
      str: Regresyon modelinin matematiksel gösterimi.
  """

  equation = "y = "
  # Kesme noktası değerini ekleyin
  if abs(intercept) > 1e-6:
    if intercept > 0:
      equation += " + "
    equation += f"{intercept:.4f}"
  # Katsayıları ve terimleri eşleştirin
  for term, coefficient in zip(terms, coefficients):
    # Katsayı sıfırdan farklıysa terimi ekleyin
    if abs(coefficient) > 1e-6:
      if coefficient > 0:
        equation += " + "
      equation += f"{coefficient:.4f} * {term}"



  return equation

def make_prediction(X, y, indices, degree, user_input):
    # Polynomial özellikleri ekleyin
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    X_poly_user = poly_features.fit_transform(user_input)
    cv = calculate_k(len(X))
    # Polynomial özelliklerin sütun isimlerini alın
    poly_feature_names = poly_features.get_feature_names_out(input_features=X.columns)

    # Modelin indekslerine göre terimleri belirleyin
    selected_terms = [poly_feature_names[i-1] for i in indices]
    indices = [i - 1 for i in indices]
    
    # Sadece seçilen terimlere sahip yeni bir matris oluşturun
    selected_X_poly = X_poly[:, indices]
    selected_X_poly_user = X_poly_user[:, indices]

    # Modeli oluşturun
    model = LinearRegression()

    # Cross-validation için pipeline oluşturun
    pipeline = make_pipeline(poly_features, model)


    # Modeli eğitin
    model.fit(selected_X_poly, y)

    # Tahminleri alın
    y_pred = model.predict(selected_X_poly_user)
    # Negatif değerleri sıfıra dönüştür
    y_pred[y_pred < 0] = 0

    return y_pred





def calculate_bootstrap_ci(X, y, n_iterations=1000):
    bootstrap_predictions = []

    for _ in range(n_iterations):
        indices = np.random.choice(len(X), len(X), replace=True)
        X_bootstrap = X.iloc[indices]
        y_bootstrap = y.iloc[indices]

        model_bootstrap = LinearRegression()
        model_bootstrap.fit(X_bootstrap, y_bootstrap)

        # NaN değer içeren sütunları çıkar
        X_bootstrap_cleaned = X_bootstrap.dropna(axis=1)

        # Eğer örneklemleme sonucunda tüm sütunlar NaN olmuşsa, o örneği atla
        if X_bootstrap_cleaned.shape[1] == 0:
            continue

        bootstrap_predictions.append(model_bootstrap.predict(X_bootstrap_cleaned))

    # Bootstrapping sonuçları üzerinden güven aralığını hesaplama
    lower_bound = np.percentile(bootstrap_predictions, 5, axis=0)
    upper_bound = np.percentile(bootstrap_predictions, 95, axis=0)

    return lower_bound, upper_bound


def plot_prediction_with_ci(y, y_pred, lower_bound, upper_bound):
    # Gerçek değerleri ve tahminleri içeren bir DataFrame oluşturun
    results_df = pd.DataFrame({
        "Gerçek Değerler": y.values,
        "Tahmin Değerleri": y_pred,
        "Alt Sınır": lower_bound,
        "Üst Sınır": upper_bound
    })

    # İndex'i sıfırla
    results_df.reset_index(drop=True, inplace=True)

    # Grafik çizimi
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df.index, results_df["Gerçek Değerler"], label="Gerçek Değerler", marker='o')
    ax.plot(results_df.index, results_df["Tahmin Değerleri"], label="Tahmin Değerleri", marker='o')
    ax.fill_between(results_df.index, results_df["Alt Sınır"], results_df["Üst Sınır"], color='gray', alpha=0.3, label="Güven Aralığı")
    ax.set_xlabel("Gözlem İndexleri")
    ax.set_ylabel("Değerler")
    ax.set_title("Tahmin Değerleri ve Güven Aralığı")
    ax.legend()

    # Streamlit üzerinde grafik gösterimi
    st.pyplot(fig)

def klasor_islemleri(degisken_deger):
    # Klasör adını belirt
    klasor_ad = "combo"

    # Klasörü oluştur
    if not os.path.exists(klasor_ad):
        os.makedirs(klasor_ad)

    # Klasördeki tüm dosyaları sil
    dosyalar = os.listdir(klasor_ad)
    for dosya in dosyalar:
        dosya_yolu = os.path.join(klasor_ad, dosya)
        if os.path.isfile(dosya_yolu):
            os.remove(dosya_yolu)

    # SQLite veritabanı bağlantısı oluştur
    veritabani_yolu = os.path.join(klasor_ad, "combo.db")
    baglanti = sqlite3.connect(veritabani_yolu)
    imlec = baglanti.cursor()

    # Değişkeni kaydet
    imlec.execute("CREATE TABLE IF NOT EXISTS combo (degisken TEXT)")
    imlec.execute("INSERT INTO combo (degisken) VALUES (?)", (degisken_deger,))

    # Veritabanı değişikliklerini kaydet ve bağlantıyı kapat
    baglanti.commit()
    baglanti.close()

def degeri_oku_ve_yazdir():
    # SQLite veritabanı bağlantısı oluştur
    klasor_ad = "combo"
    veritabani_yolu = os.path.join(klasor_ad, "combo.db")
    baglanti = sqlite3.connect(veritabani_yolu)
    imlec = baglanti.cursor()

    # Değerleri oku
    imlec.execute("SELECT * FROM combo")
    veriler = imlec.fetchall()

    # Bağlantıyı kapat
    baglanti.close()
    return veriler
