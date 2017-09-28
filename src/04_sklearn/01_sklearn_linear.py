import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([0, 1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([5, 7, 9, 11, 13, 15]).reshape(-1, 1)

reg = LinearRegression()
reg.fit(x, y)

print(reg.predict([[6], [7], [8]]))

print(reg.coef_)
print(reg.intercept_)
