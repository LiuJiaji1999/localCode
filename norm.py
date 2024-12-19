import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 创建子图
fig, ax = plt.subplots(1, 1)
# 定义正态分布
norm_0 = norm(loc=0, scale=1)
norm_1 = norm(loc=1, scale=2)
# 定义横坐标范围
x = np.linspace(-10, 10, 1000)  # 修正为从 -10 到 10
# 绘制概率密度函数 (PDF)
ax.plot(x, norm_0.pdf(x), color='red', lw=3, alpha=0.6, label='loc=0, scale=1')
ax.plot(x, norm_1.pdf(x), color='blue', lw=3, alpha=0.6, label='loc=1, scale=2')
# 添加图例
ax.legend(loc='best', frameon=False)
# # 添加网格线
# plt.grid(ls='--')
# # 显示图像
# plt.show()

norm_rv = norm(loc=1, scale=2)
norm_rvs = norm_rv.rvs(size=100000) # 10 万次采样
x = np.linspace(-10, 10, 1000)
plt.plot(x, norm_rv.pdf(x), 'r', lw=3, alpha=0.6, label="$\\mu$=2,$\\sigma$=2")
plt.hist(norm_rvs, density=True, bins=50, alpha=0.6,edgecolor='k')
plt.legend()
plt.grid(ls='--')
plt.show()
