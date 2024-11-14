import os
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

if __name__ == '__main__':
    os.chdir(r'D:\pycharm\data\网络谣言的文本信息识别与分析\网络谣言的文本信息识别与分析')
    file_path = "False_filter_tokenize_data_cleaned.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        documents = file.readlines()

    # 清理文档数据：去除换行符
    documents = [document.strip() for document in documents]

    # 将文档转换为词袋表示
    texts = [[word for word in document.split()] for document in documents]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 计算不同主题数下的一致性
    coherence_values = []
    model_list = []
    for num_topics in range(2, 7):
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=50)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_values.append(coherence_lda)
        model_list.append(lda_model)

    # 绘制一致性图
    plt.plot(range(2, 7), coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Coherence Score vs Number of Topics")
    plt.show()

    # 可视化最优模型
    best_model_index = coherence_values.index(max(coherence_values))
    best_model = model_list[best_model_index]
    vis = gensimvis.prepare(best_model, corpus, dictionary)
    pyLDAvis.display(vis)

    pyLDAvis.save_html(vis, 'lda_visualization3.html')