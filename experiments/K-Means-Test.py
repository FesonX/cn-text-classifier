# -*- coding: utf-8 -*-
"""
K-means
"""

from sklearn.cluster import KMeans
from tools.preprocess import *
from tools.visualizer import plot_result
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
from tools.labelText import LabelText
import settings
import time

"""
loading source
"""
print('------Loading Source...')
# sentences = []
# # content_lines = loading_source(settings.SOURCE_DATA + 'source.csv')
# # cut_source(content_lines, sentences)
#
# piracy_lines = loading_source(settings.SOURCE_DATA + 'Piracy-301.csv')
# medical_lines = loading_source(settings.SOURCE_DATA + 'Medical-1006.csv')
# game_lines = loading_source(settings.SOURCE_DATA + 'Game-397.csv')
# edu_lines = loading_source(settings.SOURCE_DATA + 'Edu-Invest-838.csv')
#
# start = time.time()
#
# cut_source(piracy_lines, sentences)
# cut_source(medical_lines, sentences)
# cut_source(game_lines, sentences)
# cut_source(edu_lines, sentences)
#
# end = time.time()
# print('------- cutting cost', end - start)
#
# ori_path = write_into_file(sentences)

ori_path = settings.SOURCE_DATA + 'cutText.csv'
sentences = loading_source(file_name=ori_path)


"""
Vertorizer
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
weight = tfidf.toarray()

end = time.time()

print("Shape: Documents(Class) / Words")
print(weight.shape)

print('------ vectorizer cost', end-start)


"""
Compute K-Means
"""

start = time.time()
numOfClass: int = 4
pca = PCA(n_components=10)
trainingData = pca.fit_transform(weight)

clf = KMeans(n_clusters=numOfClass, max_iter=10000, init="k-means++", tol=1e-6)

result = clf.fit(trainingData)
source = list(clf.predict(trainingData))
end = time.time()

print('-------training cost', end - start)
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

labelAndText = LabelText(label, ori_path)
labelAndText.sortByLabel(write=True)

plot_result(trainingData, source, numOfClass)



