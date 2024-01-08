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

#------------------------------------------------------------------------------------------
def polynomial_regression_analysis(X, y, degree, feature_names=None):
    # Create a PolynomialFeatures object to transform the data
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Get the feature names for the transformed data
    if feature_names is None:
        # Calculate feature names based on the number of columns in X_poly
        feature_names = poly.get_feature_names_out(input_features=X.columns)

    # Create a DataFrame with the polynomial features
    column_names = poly.get_feature_names_out(input_features=X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=column_names)

    # Fit a polynomial regression model with selected features
    if feature_names is not None:
        X_poly_selected = X_poly_df[feature_names]
    else:
        X_poly_selected = X_poly_df

    model = LinearRegression()
    model.fit(X_poly_selected, y)

    # Calculate R-squared
    y_pred = model.predict(X_poly_selected)
    r2 = calculate_r_squared(y, y_pred)

    # Calculate p-values
    p_values = calculate_p_values(model, X_poly_selected, y)
    
    coefficients = model.coef_

    # Create a dictionary associating feature names with p-values
    feature_p_values = dict(zip(feature_names, p_values))

    results = {
        "R-squared": r2,
        "P-values": feature_p_values,
        "coefficients": coefficients
    }

    return results



#--------------------------------------------------------------------------------------------------

def find_best_polynomial_combinations(X, y, top_n=10, max_degree=3, max_terms=9):
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

            if all(p <= 0.05 for p in p_values) and all(p <= 2  for p in vif_data["VIF"]):
                all_combinations.append((combo, r2, terms, vif_values,feature_p_values))

    all_combinations.sort(key=lambda x: x[1], reverse=True)
    top_combinations = all_combinations[:top_n]

    return top_combinations


 # Example usage:
df = pd.read_csv("SAMBVK.csv", delimiter=";")
X = df[['x1', 'x3']]
X = standardize(X)
y = df['y']
top_10_combinations = find_best_polynomial_combinations(X, y, max_terms=15, top_n=5)

for i, (combo, r2, terms,vif_values,feature_p_values) in enumerate(top_10_combinations):
    print("--------------------------")
    print(combo)
    print("--------------------------")
    print(terms)
    for feature, p_value in vif_values.items():
        print(f"{feature}: {p_value}")
    print("--------------------------")
    for feature, p_value in feature_p_values.items():
        print(f"{feature}: {p_value}")
    print("--------------------------")
    print("R-squared:", r2)
    print("--------------------------")
    print("##################################################")

  

 
