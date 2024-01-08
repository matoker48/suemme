import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(a, b, c, d, e, x1, x2):
    return a + b * x1 + c * x2 + d * x1 * x2 + e * (x1**2) * x2

def obj_fun(ps, x1, x2, y):
    return np.sum((f(ps[0], ps[1], ps[2], ps[3], ps[4], x1, x2) - y)**2)

def fun_residuals(params, x1, x2, y):
    return f(params[0], params[1], params[2], params[3], params[4], x1, x2) - y

def plot_regression(XD, Yhat, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x1fit = np.linspace(min(XD[:, 1]), max(XD[:, 1]), 20)
    x2fit = np.linspace(min(XD[:, 2]), max(XD[:, 2]), 20)
    X1FIT, X2FIT = np.meshgrid(x1fit, x2fit)
    YFIT = f(Yhat[0], Yhat[1], Yhat[2], Yhat[3], Yhat[4], X1FIT, X2FIT)

    ax.plot_surface(X1FIT, X2FIT, YFIT, edgecolor='k', alpha=0.7)
    ax.scatter(XD[:, 1], XD[:, 2], XD[:, 0], c='r', marker='o')

    ax.set_xlabel('Sinterleme Sıcaklığı')
    ax.set_ylabel('Tane Boyut')
    ax.set_zlabel('Su Emme Miktarı')
    ax.set_title(title)

    return fig

# Veri Seti Tanımlamaları
df = pd.read_csv("SAMBVK.csv", delimiter=";")
XDyu = df[['y', 'x1', 'x3']].values
XDy = df[['y', 'x1', 'x3']].values

# Normalizasyon
Xy = XDy.T
Xy = (Xy - np.mean(Xy, axis=1, keepdims=True)) / np.std(Xy, axis=1, keepdims=True)

# Katsayı Ayarı
ck = 1
XD = np.zeros_like(XDy)
XD[:, 0] = XDy[:, 0]
XD[:, 1] = ck * XDy[:, 1]
XD[:, 2] = XDy[:, 2]

asd = len(XD)
X = np.vstack([np.ones(asd), XD[:, 1], XD[:, 2], XD[:, 1] * XD[:, 2], (XD[:, 1]**2) * XD[:, 2]])

# Doğrusal Regresyon
bhat = np.linalg.lstsq(X.T, XD[:, 0], rcond=None)[0]

# Nonlinear Optimizasyon
x0 = bhat
sol = minimize(obj_fun, x0, args=(XD[:, 1], XD[:, 2], XD[:, 0]), method='Nelder-Mead')

# Robust Regresyon
robust_params = least_squares(fun_residuals, x0, loss='soft_l1', f_scale=0.1, args=(XD[:, 1], XD[:, 2], XD[:, 0]))

# Streamlit Uygulaması
st.title('Regresyon Analizi')

# Doğrusal Regresyon Grafiği
st.subheader('Doğrusal Regresyon')
linear_plot = plot_regression(XD, bhat, 'Doğrusal Regresyon')
st.pyplot(linear_plot)

# Nonlinear Regresyon Grafiği
st.subheader('Nonlinear Regresyon')
nonlinear_plot = plot_regression(XD, sol.x, 'Nonlinear Regresyon')
st.pyplot(nonlinear_plot)

# Robust Regresyon Grafiği
st.subheader('Robust Regresyon')
robust_plot = plot_regression(XD, robust_params.x, 'Robust Regresyon')
st.pyplot(robust_plot)
