# -*- coding: utf-8 -*-
"""
Birch
"""

from sklearn.cluster import Birch
from tools.preprocess import *
from tools.visualizer import plot_result
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from tools.labelText import LabelText
import settings
import time
import pandas as pd
from sklearn import metrics

"""
loading source
载入资源
文件详情参照本文件夹README
"""
print('------Loading Source...')
ori_path = settings.SOURCE_DATA + 'cut_data.csv'
sentences = loading_source(file_name=ori_path)
# content_lines = loading_source(file_name=ori_path)
# start = time.time()
# cut_source(content_lines, sentences)
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
pca = PCA(n_components=10)
trainingData = pca.fit_transform(weight)
# svd = TruncatedSVD(n_components=10, n_iter=10, random_state=42)
# trainingData = svd.fit_transform(weight)


"""
Compute Birch
"""

numOfClass: int = 4

start = time.time()
clf = Birch(n_clusters=4, branching_factor=10, threshold=0.01)

result = clf.fit(trainingData)
source = list(clf.predict(trainingData))
end = time.time()

label = clf.labels_

labelAndText = LabelText(label, ori_path)
labelAndText.sortByLabel(show=False, write=True)


"""
Result
生成各个指标并写入文件
"""
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
    w.write("-------Birch Experiment-------\n")
    w.write("adjusted_rand_score: %f\n" % ars)
    w.write("FMI: %f\n" % fmi)
    w.write("Silhouette: %f\n " % silhouette)
    w.write("CHI: %f\n" % CHI)
    w.write("------End------")

plot_result(trainingData, source, numOfClass)
