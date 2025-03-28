import pandas as pd
import matplotlib.pyplot as plt

# 创建DataFrame
data = {
    'Epoch': list(range(1, 21)),
    'Ours': [
        84.17786362, 87.71138775, 87.32139899, 88.55979583, 90.14052969,
        87.36096941, 90.54907887, 90.69524926, 90.66612677, 91.02081253,
        91.02387424, 91.62032624, 91.73842921, 91.72501282, 91.34600327,
        92.3337353, 91.82829242, 91.70587682, 92.61664946, 92.32862946
    ],
    'Original model': [
        83.10312306, 83.91414635, 85.90460389, 89.08754499, 88.74896583,
        89.02976385, 90.11282582, 89.60269807, 89.93242987, 90.34993966,
        91.27649514, 91.34430717, 91.5089139, 91.43769431, 91.49250233,
        91.50927562, 91.47347318, 91.34390854, 91.50712579, 91.47039647
    ]
}
df = pd.DataFrame(data)

# 绘制箱线图
plt.figure(figsize=(10, 6))
boxplot = df.boxplot(column=['Ours', 'Original model'])
plt.xlabel('Model')
plt.ylabel('mIou')
plt.title('Model Comparison')

# 标注最大值
for i, col in enumerate(['Ours', 'Original model']):
    max_value = df[col].max()
    max_index = df[col].idxmax()
    plt.text(i + 1, max_value, f'Max: {max_value:.2f} (Epoch {max_index*5})', ha='center', va='bottom', color='red')

plt.grid(True)

# 保存图像
plt.savefig('model_comparison.png')

# 显示图像
plt.show()
