# -*- coding: utf-8 -*-

"""
Birch
"""

from sklearn.cluster import Birch
from tools.preprocess import *
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from tools.labelText import LabelText
from tools.visualizer import plot_result
import settings


"""
loading source
"""
print("Loading Sources...")
# content_lines = loading_source(settings.SOURCE_DATA + 'source.csv')
# cut_source(content_lines, sentences)
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
sentences = loading_source(settings.SOURCE_DATA + 'cutText.csv')
ori_path = settings.SOURCE_DATA + 'cutText.csv'


"""
Vertorizer
"""
print("Vertorizer...")
start = time.time()

vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
transformer = TfidfTransformer()
freq_words_matrix = vertorizer.fit_transform(sentences)
words = vertorizer.get_feature_names()
tfidf = transformer.fit_transform(freq_words_matrix)
weight = tfidf.toarray()

print("Shape: Documents(Class) / Words")
print(weight.shape)

end = time.time()
print('------ vectorizer cost', end-start)


"""
Compute Birch
"""
start = time.time()
numOfClass = 4
bc = Birch(n_clusters=4, branching_factor=10, threshold=0.01)

# Dimension Reduction
pca = PCA(n_components=10)
trainingData = pca.fit_transform(weight)

result = bc.fit(trainingData)
source = list(bc.predict(trainingData))
labels = bc.labels_
end = time.time()
print('-------training cost', end - start)

labelAndText = LabelText(labels, ori_path)
labelAndText.sortByLabel(show=False, write=True, algorithm="bc")

plot_result(trainingData, source, numOfClass)
