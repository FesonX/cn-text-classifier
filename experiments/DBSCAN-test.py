# -*- coding: utf-8 -*-

"""
DBSCAN
"""

from sklearn.cluster import DBSCAN
from tools.preprocess import *
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
from tools.labelText import LabelText
import matplotlib.pyplot as plt
import numpy as np
import settings

"""
loading source
"""
print("Loading Sources...")
sentences = []
content_lines = loading_source(settings.SOURCE_DATA + 'source.csv')
cut_source(content_lines, sentences)

"""
Vertorizer
"""
# print("Vertorizer...")
# vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
# transformer = TfidfTransformer()
#
# freq_words_matrix = vertorizer.fit_transform(sentences)
#
# words = vertorizer.get_feature_names()
#
# tfidf = transformer.fit_transform(freq_words_matrix)
#
# weight = tfidf.toarray()

# 词频矩阵 Frequency Matrix Of Words
vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.6)
transformer = TfidfTransformer()
# Fit Raw Documents
freq_words_matrix = vertorizer.fit_transform(sentences)
# Get Words Of Bag
words = vertorizer.get_feature_names()

tfidf = transformer.fit_transform(freq_words_matrix)

weight = tfidf.toarray()

print("Shape: Documents(Class) / Words")
print(weight.shape)


"""
Compute DBSCAN
"""

# pca = PCA(n_components=10)
# trainingData = pca.fit_transform(weight)

# """
# compute DBSCAN
# """
# db = DBSCAN(eps=0.36, min_samples=1)
# result = db.fit(trainingData)
# core_samples_mask = np.zeros_like(result.labels_, dtype=bool)
# core_samples_mask[result.core_sample_indices_] = True
# labels = result.labels_
#
# pca = PCA(n_components=3)
# original = pca.fit_transform(weight)
# source = list(db.fit_predict(trainingData))
# print(trainingData)
# print(source)
#
# labels = source
#
# """
# Number of cluster in labels, ignoring noise if present.
# """
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
#
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
#
#
# def plotRes(data, clusterRes, clusterNum):
#     nPoints = len(data)
#     scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
#     for i in range(clusterNum):
#         color = scatterColors[i % len(scatterColors)]
#         x1 = []
#         y1 = []
#         for j in range(nPoints):
#             if clusterRes[j] == i:
#                 x1.append(data[j, 0])
#                 y1.append(data[j, 1])
#         plt.scatter(x1, y1, c=color, alpha=1, marker='+')
#
#
# plotRes(trainingData, source, n_clusters_)
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()

db = DBSCAN(eps=0.35, min_samples=1)

pca = PCA(n_components=9)
trainingData = pca.fit_transform(weight)

original = pca.fit_transform(weight)

s = db.fit(trainingData)

label = db.fit_predict(trainingData)
print(list(label))

core_samples_mask = np.zeros_like(s.labels_, dtype=bool)
core_samples_mask[s.core_sample_indices_] = True

print("---- Clusters Of Samples: \n")
print(db.labels_)

i = 1
while i < len(db.labels_):
    print(i, db.labels_[i-1])
    i += 1

labels = db.labels_
ori_path = settings.SOURCE_DATA + 'source.csv'

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

labelAndText = LabelText(labels, ori_path)
labelAndText.sortByLabel(write=True, algorithm='db')
