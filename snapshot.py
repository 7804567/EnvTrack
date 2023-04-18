import netCDF4
import numpy as np
import gzip
import shutil

with gzip.open('RG_ArgoClim_Temperature_2019.nc.gz', 'rb') as f_in:
    with open('RG_ArgoClim_Temperature_2019.nc', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

data = netCDF4.Dataset('RG_ArgoClim_Temperature_2019.nc', format="NETCDF4")

temp_mean = np.array(data["ARGO_TEMPERATURE_MEAN"][1, :, :])
data["PRESSURE"][1]



theta = np.linspace(0, 2 * np.pi, 1000)
plt.plot(theta, np.sqrt(np.abs(np.sin(theta)) + np.abs(np.cos(theta))), label="l1")
plt.plot(theta, b2 / (1 - (e2 * np.cos(theta))), label="ellipse")
plt.legend()

plt.plot(theta, (1 + (2 * np.sin(theta) ** 2)) * (1 + (e2 * np.cos(theta) ** 2)) - 2)

b2 = 1
e2 = 0.5
from sympy import Symbol, Derivative, cos, lambdify, sin, sqrt, expand, factor, symbols, tan
k = lambda r, r1, r2: (r ** 2 + (2 * r1 ** 2) - r * r2)
f = lambda theta: np.sqrt((1+np.tan(theta)**2)/ (1+np.tan(theta))**2)
theta = Symbol('theta')
deriv0 = sqrt((1+tan(theta)**2)/ (1+tan(theta))**2)
deriv1 = Derivative(deriv0, theta, 1)
deriv2 = Derivative(deriv0, theta, 2)
f1 = lambdify(theta, deriv1.doit(), "numpy")
f2 = lambdify(theta, deriv2.doit(), "numpy")
h = np.linspace(0.1, 1, 1000)
plt.plot(h, k(f(h), f1(h), f2(h)))

plt.plot(f(h) * np.cos(h), f(h) * np.sin(h))

k = lambda r, r1, r2: (r ** 2 + (2 * r1 ** 2) - r * r2) / ((r ** 2 + r2 ** 2) ** (3 / 2))
f = lambda theta: np.sqrt(np.cos(theta) + np.sin(theta))
theta = Symbol('theta')
deriv0 = sqrt(cos(theta) + sin(theta))
deriv1 = Derivative(deriv0, theta, 1)
deriv2 = Derivative(deriv0, theta, 2)
f1 = lambdify(theta, deriv1.doit(), "numpy")
f2 = lambdify(theta, deriv2.doit(), "numpy")
h = np.linspace(0, np.pi / 2, 1000)
plt.plot(h, k(f(h), f1(h), f2(h)))

x, y, z = symbols('x y z')
expr = x ** 4 + 2 * x ** 3 + x ** 2 + (2 * (y ** 2)) - ((x + x ** 2) * (z * (1 + x) - (2 * (y ** 2))))
factor(expr)

