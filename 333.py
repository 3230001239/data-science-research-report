import os
import re
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

if __name__ == '__main__':
    rcParams['font.sans-serif'] = ['SimHei']
    os.chdir(r'D:\pycharm\data\网络谣言的文本信息识别与分析\网络谣言的文本信息识别与分析')
    path = '中间结果.csv'
    pattern = r"(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|" \
              r"Oct(ober)?|Nov(ember)?|Dec(ember)?)\s+\d+,\s+\d*"
    pattern2 = r"(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|" \
              r"Oct(ober)?|Nov(ember)?|Dec(ember)?)\s+\d+,\s+\d{4}"
    data = pd.read_csv(path, header=0)
    # print(data.head())
    data['StartDate'] = data['StartDate'].apply(lambda x: re.search(pattern, x).group(0) if re.search(pattern, x) else x)
    data['EndDate'] = data['EndDate'].apply(
        lambda x: re.search(pattern, x).group(0) if re.search(pattern, x) else x)
    # data['StartDate'].to_csv('001.csv')
    # print(data['StartDate'])
    data = data[data['Label'] == "FALSE"]
    # data.to_csv("所有谣言新闻01.csv")
    data = data[data['StartDate'].apply(lambda x: True if(re.search(pattern2, x)) else False)]
    data = data[data['EndDate'].apply(lambda x: True if (re.search(pattern2, x)) else False)]
    data['StartDate'].to_csv('001.csv')
    data.to_csv('所有谣言数据时间完成版.csv', index=False)
    df = data
    # 转换日期格式
    df['StartDate'] = pd.to_datetime(df['StartDate'])
    df['EndDate'] = pd.to_datetime(df['EndDate'])

    # 计算传播时间中点
    df['MidPointDate'] = df.apply(lambda row: row['StartDate'] + (row['EndDate'] - row['StartDate']) / 2, axis=1)

    # 添加年份和季度列
    df['Year'] = df['MidPointDate'].dt.year
    df['Quarter'] = df['MidPointDate'].dt.quarter

    # 创建年份-季度列，格式为'年份-Q季度'
    df['Year-Quarter'] = df.apply(lambda x: f"{x['Year']}-Q{x['Quarter']}", axis=1)

    # 根据年份-季度分组并计算谣言数量
    rumor_counts = df.groupby('Year-Quarter').size()

    # 绘制条形图
    plt.figure(figsize=(14, 8))
    rumor_counts.plot(kind='bar')
    plt.title('谣言数与年份季度关系图')
    plt.xlabel('年份-季度')
    plt.ylabel('谣言数')
    plt.xticks(rotation=45)  # 旋转x轴标签以便阅读
    plt.tight_layout()  # 调整布局

    # 保存图形为PNG图片
    plt.savefig('谣言传播时间图.png')