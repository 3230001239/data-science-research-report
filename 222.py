import pandas as pd
import os
import nltk
import matplotlib.pyplot as plt
import wordcloud
from itertools import chain

if __name__ == '__main__':
    file_path = 'B题数据.csv'
    os.chdir(r'D:\pycharm\data\网络谣言的文本信息识别与分析\网络谣言的文本信息识别与分析')
    data = pd.read_csv(file_path, header=0, encoding='Windows-1252')
    data = data.iloc[:, :7]
    data.dropna(axis=0, how='any', inplace=True)
    data = data[~data['Label'].str.contains(r'[.,]', regex=True)]
    # print(data['Title'])
    # data.to_csv('中间结果.csv')
    # nltk.download('punkt')
    data['tokenize'] = data['Title'].apply(lambda x : nltk.tokenize.word_tokenize(x))  #分词
    # print(data['tokenize'])  #分词
    # print(nltk.data.path)
    stopwords = pd.read_csv('baidu_stopwords.txt', header=None)
    stopwords_list = stopwords.iloc[:, 0].tolist()
    # print(stopwords_list)  #停用词列表

    def remove_stop(text, stopw):
         return [word for word in text if word.lower() not in stopw]  #对字符串进行处理
    data['filter_tokenize'] = data['tokenize'].apply(lambda x: remove_stop(x, stopwords_list))
    print(data['filter_tokenize'])  #去除停用词后的

    # 将所有的词列表展平成一个Pandas Series
    flat_series = pd.Series(list(chain.from_iterable(data['filter_tokenize'])))
    # 使用value_counts()统计词频
    word_counts = flat_series.value_counts()

    word_counts.to_csv('词频.csv')
    # data['filter_tokenize'].to_csv('11.csv')
    long_text = ' '.join([' '.join(words) for words in data['filter_tokenize']])
    # print(long_text)
    wc = wordcloud.WordCloud(background_color='white').generate(long_text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')  # 不显示坐标轴
    plt.show()
