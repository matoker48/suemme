import numpy as np
import pandas as pd
from scipy.optimize import minimize, least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
def f(a, b, c, d, e, x1, x2):
    return a + b * x1 + c * x2 + d * x1 * x2 + e * (x1**2) * x2

def obj_fun(ps):
    return np.sum((f(ps[0], ps[1], ps[2], ps[3], ps[4], XD[:, 1], XD[:, 2]) - XD[:, 0])**2)

x0 = bhat
sol = minimize(obj_fun, x0, method='Nelder-Mead')

# Robust Regresyon
def fun_residuals(params, x1, x2, y):
    return f(params[0], params[1], params[2], params[3], params[4], x1, x2) - y

robust_params = least_squares(fun_residuals, x0, loss='soft_l1', f_scale=0.1, args=(XD[:, 1], XD[:, 2], XD[:, 0]))

Yhaty = bhat[0] + bhat[1] * XD[:, 1] + bhat[2] * XD[:, 2] + bhat[3] * (XD[:, 1] * XD[:, 2]) + bhat[4] * ((XD[:, 1]**2) * XD[:, 2])

# Diğer işlemler
# ...

# Görselleştirme
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1fit = np.linspace(min(XD[:, 1]), max(XD[:, 1]), 20)
x2fit = np.linspace(min(XD[:, 2]), max(XD[:, 2]), 20)
X1FIT, X2FIT = np.meshgrid(x1fit, x2fit)
YFIT1 = f(bhat[0], bhat[1], bhat[2], bhat[3], bhat[4], X1FIT, X2FIT)

ax.plot_surface(X1FIT, X2FIT, YFIT1, edgecolor='k', alpha=0.7)

ax.scatter(XD[:, 1], XD[:, 2], XD[:, 0], c='r', marker='o')

# Robust Regresyon sonuçlarını görselleştirme
YFIT_robust = f(robust_params.x[0], robust_params.x[1], robust_params.x[2], robust_params.x[3], robust_params.x[4], X1FIT, X2FIT)
ax.plot_surface(X1FIT, X2FIT, YFIT_robust, edgecolor='b', alpha=0.7)

ax.set_xlabel('Sinterleme Sıcaklığı')
ax.set_ylabel('Tane Boyut')
ax.set_zlabel('Su Emme Miktarı')

plt.show()
