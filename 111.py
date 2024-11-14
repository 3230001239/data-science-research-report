import pandas as pd
import os
import chardet
import matplotlib.pyplot as plt

if __name__ == '__main__':
    os.chdir(r'D:\pycharm\data\网络谣言的文本信息识别与分析\网络谣言的文本信息识别与分析')

    def detect_encoding(file_path):
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取部分数据进行编码检测
            result = chardet.detect(raw_data)
            return result['encoding']
    file_path = '数据.csv'
    print(detect_encoding(file_path))
    data = pd.read_csv(file_path, header=0, encoding='Windows-1252')
    data = data.iloc[:, :7]
    data.dropna(axis=0, how='any', inplace=True)
    print(data.columns)
    print(data.head())
    counts = data['Label'].value_counts()
    print(counts)
    data = data[~data['Label'].str.contains(r'[.,]', regex=True)]
    counts = data['Label'].value_counts(normalize=True)
    print(counts)
    print(data.head())
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'yellow']
    # 绘制饼状图 数据可视化
    fig, ax = plt.subplots()
    ax.pie(counts, colors=colors, autopct='%1.1f%%', startangle=90, labels=None)

    legend_labels = [plt.Line2D([0], [0], linestyle='', marker='o', markersize=15, markerfacecolor=color, label=label)
                     for color, label in zip(colors, counts.index)]

    # 添加图例
    ax.legend(handles=legend_labels, title='Legend', loc='best', borderpad=0.001)
    ax.axis('equal')
    plt.title('proportion')

    # 保存并显示图表
    plt.savefig('词云图.png')  # 保存图表为PNG文件
    plt.show()  # 显示图表
