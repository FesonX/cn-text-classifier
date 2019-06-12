# -*- coding: utf-8 -*-
import jieba
import pandas as pd
import settings
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.decomposition import PCA
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from itertools import combinations


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
    jieba.load_userdict(settings.STATIC_DIR + 'dict.txt')
    count = 0
    if write:
        now = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        write_path = settings.DST_DATA + now + '-cut.csv'
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
                w.write('%d,' % count)
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


def snn_sim_matrix(X, k=5):
    """
    利用sklearn包中的KDTree,计算节点的共享最近邻相似度(SNN)矩阵
    :param X: array-like, shape = [samples_size, features_size]
    :param k: positive integer(default = 5), compute snn similarity threshold k
    :return: snn distance matrix
    """
    try:
        X = np.array(X)
    except Exception as e:
        print(e)
        raise ValueError("输入的数据集必须为矩阵")
    samples_size, features_size = X.shape  # 数据集样本的个数和特征的维数
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(X)
    knn_matrix = nbrs.kneighbors(X, return_distance=False)  # 记录每个样本的k个最近邻对应的索引
    sim_matrix = 0.5 + np.zeros((samples_size, samples_size))  # snn相似度矩阵
    for i in range(samples_size):
        t = np.where(knn_matrix == i)[0]
        c = list(combinations(t, 2))
        for j in c:
            if j[0] not in knn_matrix[j[1]]:
                continue
            sim_matrix[j[0]][j[1]] += 1
    sim_matrix = 1 / sim_matrix  # 将相似度矩阵转化为距离矩阵
    sim_matrix = np.triu(sim_matrix)
    sim_matrix += sim_matrix.T - np.diag(sim_matrix.diagonal())
    return sim_matrix
