# -*- coding: utf-8 -*-

"""
DBSCAN
"""

from sklearn.cluster import DBSCAN
from tools.preprocess import *
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
from tools.labelText import LabelText
from tools.visualizer import plot_result
import numpy as np
import settings

"""
loading source
"""
print("Loading Sources...")
# sentences = []
# content_lines = loading_source(settings.SOURCE_DATA + 'source.csv')
# cut_source(content_lines, sentences)
ori_path = settings.SOURCE_DATA + 'cutText.csv'
sentences = loading_source(file_name=ori_path)


"""
Vertorizer
"""
print("Vertorizer...")
vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
transformer = TfidfTransformer()
freq_words_matrix = vertorizer.fit_transform(sentences)
words = vertorizer.get_feature_names()
tfidf = transformer.fit_transform(freq_words_matrix)
weight = tfidf.toarray()

print("Shape: Documents(Class) / Words")
print(weight.shape)


"""
Compute DBSCAN
"""

# db = DBSCAN(eps=0.36, min_samples=3)
# 0.03-0.036 min_samples 8
# 0.38 噪声突然暴增
db = DBSCAN(eps=0.0355, min_samples=9)

# n=10,较好， n=11 类别过多， n=12 类别过少, n=13 噪声猛增, n=14 几乎都是噪声
pca = PCA(n_components=9)
trainingData = pca.fit_transform(weight)

original = pca.fit_transform(weight)

result = db.fit(trainingData)

label = list(db.fit_predict(trainingData))

core_samples_mask = np.zeros_like(result.labels_, dtype=bool)
core_samples_mask[result.core_sample_indices_] = True


"""
Result
"""
print("---- Clusters Of Samples: \n")
print(db.labels_)

i = 1
while i < len(db.labels_):
    print(i, db.labels_[i-1])
    i += 1

labels = db.labels_

n_clusters_ = len(set(labels))
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

labelAndText = LabelText(labels, ori_path)
labelAndText.sortByLabel(write=True, algorithm='db')

plot_result(trainingData, label, n_clusters_)
