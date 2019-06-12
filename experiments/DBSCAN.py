# -*- coding: utf-8 -*-
"""
DBSCAN
"""

from sklearn.cluster import DBSCAN
from tools.preprocess import *
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
from tools.labelText import LabelText
import settings
import time
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

"""
loading source
载入资源
文件详情参照本文件夹README
"""
print('------Loading Source...')
ori_path = settings.SOURCE_DATA + 'cut_data.csv'
sentences = loading_source(file_name=ori_path)
# start = time.time()
# end = time.time()
# print('------- cutting cost', end - start)


"""
Vertorizer
向量化
"""
print('------Vertorizer...')
start = time.time()

# 词频矩阵 Frequency Matrix Of Words
vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
transformer = TfidfTransformer()
# Fit Raw Documents
freq_words_matrix = vertorizer.fit_transform(sentences)
# Get Words Of Bag
words = vertorizer.get_feature_names()
tfidf = transformer.fit_transform(freq_words_matrix)
weight = freq_words_matrix.toarray()

end = time.time()

print("Shape: Documents(Class) / Words")
print(weight.shape)

print('------ vectorizer cost', end-start)


"""
Dimension Reduction
降维
"""
pca = PCA(n_components=8)
trainingData = pca.fit_transform(weight)


"""
Compute DBSCAN
"""

numOfClass: int = 4

start = time.time()
# db = DBSCAN(eps=0.08, min_samples=7)
db = DBSCAN(eps=0.08, min_samples=7)


result = db.fit(trainingData)
source = list(db.fit_predict(trainingData))
end = time.time()

label = db.labels_

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

labelAndText = LabelText(label, ori_path)
labelAndText.sortByLabel(show=False, write=True, algorithm="DB")


"""
Visualize
考虑到 DBSCAN 算法有检测噪声的能力，单独实现一个可视化
"""


def plot_res(labels: list, n_cluster: int, num: int):
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
    for k, col in zip(set(labels), colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        class_member_mask = (labels == k)
        xy = trainingData[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=10)
        xy = trainingData[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)
    plt.title('DBSCAN')
    plt.savefig(settings.PLOT_DIR + 'db-%d-%d.png' % (n_cluster, num))
    plt.show()


"""
Result
生成各个指标并写入文件
"""
n_clusters_ = len(set(label)) - (1 if -1 in label else 0)
n_noise_ = int(list(label).count(-1))
print('Estimated number of noise points: %d \n' % n_noise_)

content = pd.read_csv(settings.SOURCE_DATA + 'labeled_data.csv')
labels_true = content.flag.to_list()

ars = metrics.adjusted_rand_score(labels_true, label)
print("adjusted_rand_score: ", ars)

fmi = metrics.adjusted_rand_score(labels_true, label)
print("FMI: ", fmi)

silhouette = metrics.silhouette_score(trainingData, label)
print("silhouette: ", silhouette)

CHI = metrics.calinski_harabaz_score(trainingData, label)
print("CHI: ", CHI)


with open(settings.DST_DATA+time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())+'result.txt', 'w') as w:
    w.write("------DBSCAN Experiment-------\n")
    w.write("adjusted_rand_score: %f\n" % ars)
    w.write("FMI: %f\n" % fmi)
    w.write("Silhouette: %f\n" % silhouette)
    w.write("CHI: %f\n" % CHI)
    w.write('Estimated number of noise points: %d \n' % n_noise_)
    w.write("------End------")

plot_res(label, n_clusters_, n_clusters_)
