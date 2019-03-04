# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import settings
import random
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA


#############
# loading source
#############

def loading_source(file_name: str):
    source = ''.join([settings.BASE_PATH, file_name])
    source_df = pd.read_csv(source, sep=',', encoding='utf-8')
    source_df.dropna(inplace=True)
    return source_df.content.values.tolist()


def cut_source(content_lines, sentences, drop_digit=False, drop_single_char=False):
    """
    cut words and give tags on words
    :param content_lines: list
    :param sentences: empty list, for saving data with tags
    :param drop_digit: default False
    :param drop_single_char: default False
    :return:None
    """
    stop_words_path = ''.join([settings.BASE_PATH, 'stop_words.txt'])
    stop_words = pd.read_csv(stop_words_path, index_col=False, quoting=3, sep='\t', names=['stopword'], encoding='utf-8')
    stop_words = stop_words['stopword'].values
    for line in content_lines:
        try:
            # cut word
            segs = jieba.lcut(line)
            if drop_digit:
                # drop digit
                segs = [s for s in segs if not str(s).isdigit()]
            # strip space
            segs = list(filter(lambda x: x.strip(), segs))
            if drop_single_char:
                # drop single char
                segs = list(filter(lambda x: len(x) > 1, segs))
            # drop stop words
            segs = list(filter(lambda x: x not in stop_words, segs))
            sentences.append(' '.join(segs))
            random.shuffle(sentences)
        except Exception as e:
            print(e)
            print(line)
            continue


def extract_characters(sentences: list, dimension: int):
    """
    vertorizer
    :param sentences: list
    :param dimension: int
    :return: weight, training_data
    """
    # Transfer into frequency matrix a[i][j], word j in text class i frequency
    vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
    # collect tf-idf weight
    transformer = TfidfTransformer()
    # outer transform for calculate tf-idf, second for transform into matrix
    tfidf = transformer.fit_transform(vertorizer.fit_transform(sentences))
    # get all words in BOW
    words_bag = vertorizer.get_feature_names()
    # w[i][j] represents word j's weight in text class i
    weight = tfidf.toarray()
    #
    print('Features length:' + str(len(words_bag)))
    pca = PCA(n_components=dimension)
    training_data = pca.fit_transform(weight)
    return weight, training_data
