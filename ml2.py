import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV, train_test_split
def calculate_k(observation_count):
    if observation_count < 200:
        result = 5
    else:
        result = observation_count // 40
    
    return int(result)
def standardize(X):
    # Standart scaler'ı oluştur
    scaler = StandardScaler()

    # Standartlaştır
    standardized_X = scaler.fit_transform(X)

    # DataFrame'e çevir
    standardized_X = pd.DataFrame(standardized_X, columns=X.columns)

    return standardized_X

def train_random_forest(X_train, y_train, param_grid):
    # Random Forest Regressor
    rf = RandomForestRegressor()
    c = calculate_k(len(X_train))
    # GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=c, scoring='r2')
    grid_search.fit(X_train, y_train)

    # En iyi parametreler
    best_params = grid_search.best_params_
    print("En iyi parametreler:", best_params)

    # En iyi modeli seçme
    best_rf = grid_search.best_estimator_

    # Train setinde R2 skoru
    r2_train = r2_score(y_train, best_rf.predict(X_train))
    print(f"Train setinde R2 Skoru: {r2_train}")

    return best_rf, r2_train

def train_decision_tree(X_train, y_train, param_grid):
    # Decision Tree Regressor
    dt = DecisionTreeRegressor()
    c = calculate_k(len(X_train))
    # GridSearchCV
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=c, scoring='r2')
    grid_search.fit(X_train, y_train)

    # En iyi parametreler
    best_params = grid_search.best_params_
    print("En iyi parametreler:", best_params)

    # En iyi modeli seçme
    best_dt = grid_search.best_estimator_

    # Train setinde R2 skoru
    r2_train = r2_score(y_train, best_dt.predict(X_train))
    print(f"Train setinde R2 Skoru: {r2_train}")

    return best_dt, r2_train

def train_svr(X_train, y_train, param_grid):
    # SVR Regressor
    svr = SVR()
    c = calculate_k(len(X_train))
    # GridSearchCV
    grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=c, scoring='r2')
    grid_search.fit(X_train, y_train)

    # En iyi parametreler
    best_params = grid_search.best_params_
    print("En iyi parametreler:", best_params)

    # En iyi modeli seçme
    best_svr = grid_search.best_estimator_

    # Train setinde R2 skoru
    r2_train = r2_score(y_train, best_svr.predict(X_train))
    print(f"Train setinde R2 Skoru: {r2_train}")

    return best_svr, r2_train

def evaluate_svr(model, X_val, y_val):
    # Validation setinde R2 skoru
    r2_val = r2_score(y_val, model.predict(X_val))
    print(f"Validation setinde R2 Skoru: {r2_val}")

    # Validation setinde MAE skoru
    mae_val = mean_absolute_error(y_val, model.predict(X_val))
    print(f"Validation setinde MAE: {mae_val}")

    # Test seti için y_pred hesapla
    y_pred_test = model.predict(X_val)
    y_pred_test[y_pred_test < 0] = 0


    return r2_val, mae_val, y_pred_test

def predict_with(model, user_input):
    check_is_fitted(model)
    # Predict using the SVR model
    prediction = model.predict(user_input)

    return prediction[0]