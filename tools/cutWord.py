# -*- coding: utf-8 -*-
import jieba
import os


stopwords_path = os.path.normpath(os.path.dirname(__file__)) + "/stopwords.txt"


class WordCut(object):
    def __init__(self, stopwords_path=stopwords_path):
        """
        :stopwords_path: 停用词文件路径

        """
        self.stopwords_path = stopwords_path

    def addDictionary(self, dict_list):
        """
        添加用户自定义字典列表
        """
        map(lambda x: jieba.load_userdict(x), dict_list)

    def seg_sentence(self, sentence, stopwords_path=None):
        """
        对句子进行分词
        """
        # print "now token sentence..."
        if stopwords_path is None:
            stopwords_path = self.stopwords_path

        def stopwordslist(filepath):
            """
            创建停用词list ,闭包
            """
            stopwords = [line.decode('utf-8').strip() for line in open(filepath, 'r').readlines()]
            return stopwords
        sentence_seged = jieba.cut(sentence.strip())
        stopwords = stopwordslist(stopwords_path)  # 这里加载停用词的路径
        outstr = ''  # 返回值是字符串
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr

    def seg_file(self, path, show=True, write=False):
        """
        对文本进行分词
        """
        print("now token file...")
        if write is True:
            write_path = '/'.join(path.split('/')[:-1]) + '/token.txt'
            w = open(write_path, 'w')
        with open(path, 'r') as p:
            for line in p.readlines():
                line_seg = self.seg_sentence(line)
                # lines_list.append(line_seg)
                if show is True:
                    print(line_seg)
                if write is True:
                    w.write(line_seg.encode('utf-8'))
                    w.write('\n')
        if write is True:
            w.close()



