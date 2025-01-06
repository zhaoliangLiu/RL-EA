import matplotlib.pyplot as plt

# 示例数据
data1 = [10, 13, 20, 17]
data2 = [40, 37, 30, 33]

# 确保每一对元素之和为50
assert all(data1[i] + data2[i] == 50 for i in range(len(data1)))

# 创建一个新的图表
plt.figure()

# 绘制data1和data2的折线图
plt.plot(data1, label='data1', marker='o')
plt.plot(data2, label='data2', marker='o')

# 标注每个点的值
for i in range(len(data1)):
    plt.text(i, data1[i], str(data1[i]), fontsize=12, ha='right')
    plt.text(i, data2[i], str(data2[i]), fontsize=12, ha='right')

# 添加标题和标签
plt.title('Data1 and Data2 Visualization')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

# 显示图表
plt.show()