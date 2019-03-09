# -*- coding: utf-8 -*-
"""
K-means
"""

import codecs
from sklearn.cluster import KMeans
from tools.preprocess import *
from tools.visualizer import plot_result
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from tools.labelText import LabelText
import settings


numOfClass: int = 5
clf = KMeans(n_clusters=numOfClass, max_iter=10000, init="k-means++", tol=1e-6)

sentences = []
content_lines = loading_source(settings.SOURCE_DATA + 'source.csv')
cut_source(content_lines, sentences)

# 词频矩阵 Frequency Matrix Of Words
vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
transformer = TfidfTransformer()
# Fit Raw Documents
freq_words_matrix = vertorizer.fit_transform(sentences)
# Get Words Of Bag
words = vertorizer.get_feature_names()

tfidf = transformer.fit_transform(freq_words_matrix)

weight = tfidf.toarray()

print("Shape: Documents(Class) / Words")
print(weight.shape)

res_path = settings.DST_DATA + 'tfidf_text.txt'
with codecs.open(res_path, 'w', 'utf-8') as result:
    # Save words split by space
    for j in range(len(words)):
        result.write(words[j] + ' ')
    result.write('\r\n\r\n')

    for i in range(len(weight)):
        for j in range(len(words)):
            result.write(str(weight[i][j]) + ' ')
        result.write('\r\n\r\n')

s = clf.fit(weight)
print("---- Central Points Of Samples: \n")
print(clf.cluster_centers_)
print("---- Clusters Of Samples: \n")
print(clf.labels_)

i = 1
while i < len(clf.labels_):
    print(i, clf.labels_[i-1])
    i += 1

print("---- Evaluate Number Of Clusters: \n")
print(clf.inertia_)

label = clf.labels_
ori_path = "/media/feson/DATA/data/source.csv"

labelAndText = LabelText(label, ori_path)
labelAndText.sortByLabel(write=True)

# pca = PCA(n_components=16)
# original = pca.fit_transform(weight)
#
# clf = KMeans(n_clusters=numOfClass, max_iter=10000, init="k-means++", tol=1e-6)
# clf.fit(original)
#
# ori_path = "/media/feson/DATA/data/source.csv"
# labelAndText = LabelText(clf.labels_, ori_path)
# labelAndText.sortByLabel(write=True)

# weight, training_data = extract_characters(sentences, 16)
#
# result = clf.fit(training_data)
#
# pca = PCA(n_components=16)
# original = pca.fit_transform(weight)
# source = list(clf.predict(training_data))
# plot_result(original, source, numOfClass)


