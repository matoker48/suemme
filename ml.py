import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils.validation import check_is_fitted

def evaluate_random_forest(X, y, cv=5):
    # Define KFold with specified number of splits
    kf = KFold(n_splits=cv, shuffle=True, random_state=1234)

    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=500, criterion="squared_error", max_depth=5)
    # Fit the rf model on your training data
    rf.fit(X, y)

    # Check if the rf model is fitted
    check_is_fitted(rf)

    # Cross-validated predictions
    y_pred = cross_val_predict(rf, X, y, cv=kf)

    # Calculate R2 score
    r2 = r2_score(y, y_pred)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)

    return rf, r2, mae, y_pred

def evaluate_decision_tree(X, y, cv=5):
    # Define KFold with specified number of splits
    kf = KFold(n_splits=cv, shuffle=True, random_state=1234)

    # Initialize Decision Tree Regressor
    dt = DecisionTreeRegressor(max_depth=5)
    # Fit the dt model on your training data
    dt.fit(X, y)

    # Check if the dt model is fitted
    check_is_fitted(dt)

    # Cross-validated predictions
    y_pred = cross_val_predict(dt, X, y, cv=kf)

    # Calculate R2 score
    r2 = r2_score(y, y_pred)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)

    return dt, r2, mae, y_pred

def evaluate_svr(X, y, cv=5):
    # Define KFold with specified number of splits
    kf = KFold(n_splits=cv, shuffle=True, random_state=1234)

    # Initialize Support Vector Regressor with a linear kernel
    svr = SVR(kernel='linear')
    # Fit the SVR model on your training data
    svr.fit(X, y)

    # Check if the SVR model is fitted
    check_is_fitted(svr)

    X_scaled = X

    # Cross-validated predictions
    y_pred = cross_val_predict(svr, X_scaled, y, cv=kf)

    # Calculate R2 score
    r2 = r2_score(y, y_pred)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y, y_pred)

    return svr, r2, mae, y_pred

def predict_with(model, user_input):
    check_is_fitted(model)
    # Predict using the SVR model
    prediction = model.predict(user_input)

    return prediction[0]