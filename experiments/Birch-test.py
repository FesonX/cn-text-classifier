# -*- coding: utf-8 -*-

"""
Birch
"""

from sklearn.cluster import Birch
from tools.preprocess import *
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from tools.labelText import LabelText
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
Compute Birch
"""
numOfClass = 4
bc = Birch(n_clusters=5, branching_factor=10)
result = bc.fit(weight)
labels = bc.labels_
ori_path = settings.SOURCE_DATA + 'source.csv'

labelAndText = LabelText(labels, ori_path)
labelAndText.sortByLabel(write=True, algorithm="bc")
