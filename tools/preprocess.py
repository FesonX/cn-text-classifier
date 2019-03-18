# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import settings
import random
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
import time


#############
# loading source
#############

def loading_source(file_name: str)->list:
    """
    loading file from given parameter
    :param file_name: absolute or relative file path
    :return: list
    """
    print("Loading File...")
    source_df = pd.read_csv(file_name, sep=',', encoding='utf-8')
    source_df.dropna(inplace=True)
    return source_df.content.values.tolist()


def cut_source(content_lines, sentences, drop_digit=False, drop_single_char=False, write=False):
    """
    cut words and give tags on words
    :param content_lines: list
    :param sentences: empty list, for saving data with tags
    :param drop_digit: default False
    :param drop_single_char: default False
    :param write: default False, write down the cutting words result into file
    :return:None
    """
    print("Cutting source...")
    start = time.time()
    stop_words_path = ''.join([settings.STATIC_DIR, 'stopwords.txt'])
    stop_words = pd.read_csv(stop_words_path, index_col=False, quoting=3, sep='\t', names=['stopword'], encoding='utf-8')
    stop_words = stop_words['stopword'].values
    count = 0
    if write:
        now = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        write_path = settings.DST_DATA + now + '-cut.txt'
        w = open(write_path, 'w')
    for line in content_lines:
        count += 1
        if count % 30 == 0:
            print('Cutting %d lines...' % count)
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
            if write:
                w.write(' '.join(segs))
                w.write('\n')
            sentences.append(' '.join(segs))
            # random.shuffle(sentences)
        except Exception as e:
            print(e)
            print(line)
            continue
    if write:
        w.close()
    end = time.time()
    print('--- cost', end - start)


def write_into_file(sentences: list)->str:
    """
    Write cut text into file
    :param sentences: list, text that already cut
    :return: str: filename
    """
    now = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    write_path = settings.DST_DATA + now + '-cut.csv'
    print('--- Cut Text is written in', write_path)
    with open(write_path, 'w') as w:
        count = 1
        w.write('index,content\n')
        for i in sentences:
            w.write(str(count) + ',')
            w.write(i)
            w.write('\n')
            count += 1
    print('--- File Already Written.')
    return write_path


def extract_characters(sentences: list, dimension: int):
    """
    vertorizer
    :param sentences: list
    :param dimension: int
    :return: weight, training_data
    """
    print("Vetorizier...")
    # Transfer into frequency matrix a[i][j], word j in text class i frequency
    vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
    # vertorizer = CountVectorizer()
    # collect tf-idf weight
    transformer = TfidfTransformer()
    # outer transform for calculate tf-idf, second for transform into matrix
    tfidf = transformer.fit_transform(vertorizer.fit_transform(sentences))
    # get all words in BOW
    words_bag = vertorizer.get_feature_names()
    # w[i][j] represents word j's weight in text class i
    weight = tfidf.toarray()
    print('Features length:' + str(len(words_bag)))
    pca = PCA(n_components=dimension)
    training_data = pca.fit_transform(weight)
    return weight, training_data
