import pandas as pd
import matplotlib.pyplot as plt

# 从Excel文件中读取数据
df = pd.read_excel('C:/Users/ASUS/Desktop/miou.xlsx', index_col=0)  # 假设文件名为 merged_data.xlsx

# 获取算法的名称列表
algorithms = df.columns

# 绘制损失值随迭代次数的变化曲线图
plt.figure(figsize=(10, 5))

for algo in algorithms:
    iterations = df.loc[:, algo].index.to_numpy()  # 将索引转换为 NumPy 数组
    loss_values = df.loc[:, algo].values
    plt.plot(iterations, loss_values, label=algo)

plt.xlabel('Iterations')
plt.ylabel('Miou')
plt.title('Miou vs. Iterations')
plt.legend()
plt.grid(True)
plt.savefig('C:/Users/ASUS/Desktop/miou.png', format='png')
plt.show()
