import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y2 = np.sin(x)
y = np.cos(x)

plt.plot(x, y, label="sin(x)")
plt.plot(x, y2, label="cos(x)", linestyle="--")
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('title')
plt.show()
# 这里虚线就是二阶导数