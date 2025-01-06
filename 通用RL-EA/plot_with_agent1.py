import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# def plot_data(data1, data2, rate = 0.8):
#     # 确保每一对元素之和相等
#     assert len(data1) == len(data2), "data1 和 data2 长度不相等"
#     total_sum = data1[0] + data2[0]
#     assert all(data1[i] + data2[i] == total_sum for i in range(len(data1))), "每对元素之和不相等"
    
#     bins = int(rate * len(data1))
#     # 创建数据框
#     data = np.vstack((np.arange(len(data1)), data2, data1)).T
#     cols = ['Index', 'data1', 'data2']

#     data_1 = pd.DataFrame({
#         'x': data[:, 0].astype(int),
#         'y': data[:, 2].astype(int),
#         'h': [cols[2]] * len(data),
#     })
#     data_2 = pd.DataFrame({
#         'x': data[:, 0].astype(int),
#         'y': data[:, 1].astype(int),
#         'h': [cols[1]] * len(data),
#     })

#     colors = dict([(h, c) for h, c in zip(cols[1:3], sns.color_palette(n_colors=2))])
#     fig, ax = plt.subplots(figsize=(9, 5))
#     sns.histplot(data_1, x='x', weights='y', hue='h', palette=colors,
#                  kde=True, bins=bins, stat='count', ax=ax, element="step", common_norm=True)

#     ylim_value = data1[0] + data2[0] + 0.05 * (data1[0] + data2[0])
#     ax.set_ylim([0, ylim_value])
#     ax.set_xlabel(cols[0])
#     ax.set_yticks([])
#     ax.set_yticklabels([])
#     ax.set_ylabel(cols[1])
#     ax.yaxis.label.set_color(colors[cols[1]])
    


#     ax2 = ax.twinx()
#     sns.histplot(data_2, x='x', weights='y', hue='h', palette=colors,
#                  kde=True, bins=bins, stat='count', ax=ax2, element="step", legend=True)
#     ax2.set_ylim([0, ylim_value])
#     ax2.invert_yaxis()
#     ax2.set_yticks([])
#     ax2.set_yticklabels([])
#     ax2.set_ylabel(cols[2])
#     ax2.yaxis.label.set_color(colors[cols[2]])
    
#     handles1 = ax.get_legend().legend_handles
#     handles2 = ax2.get_legend().legend_handles
#     ax.get_legend().remove()
#     ax2.get_legend().remove()
#     ax2.legend(handles1 + handles2, cols[1:3], title='E.v.E')
#     plt.show()
 
def plot_data(data1, data2, rate=1.0):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate histogram bins
    bins = np.arange(min(min(data1), min(data2)), max(max(data1), max(data2)) + 2, 1)
    
    # Plot data1 above axis
    ax.hist(data1, bins=bins, alpha=0.7, color='skyblue', label='Data 1')
    
    # Plot data2 below axis (multiply by -1 to flip)
    ax.hist(data2, bins=bins, alpha=0.7, color='lightcoral', 
            weights=np.ones(len(data2)) * -1 * rate, label='Data 2')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add labels
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Value')
    ax.legend()
    
    # Adjust y-axis labels to show absolute values
    yticks = ax.get_yticks()
    ax.set_yticklabels([abs(int(y)) for y in yticks])
    
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':  
    # 示例数据
    data1 = np.round(25 + 15 * np.sin(np.linspace(0, 10, 1000))).astype(int)
    data2 = 50 - data1

    # data1 = [10, 13, 20, 17, 15, 18, 21, 19, 25, 30]
    # data2 = [40, 37, 30, 33, 35, 32, 29, 31, 25, 20]

    print(data1)
    print(data2)
    # 调用函数绘图
    plot_data(data1, data2,rate=0.8)
   