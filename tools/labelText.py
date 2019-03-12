# -*- coding: utf-8 -*-
"""
将结果展示成label + text 的形式，易于观察，
并且提供输出到文档的功能
"""
import numpy as np
import settings
from tools.preprocess import loading_source
import time


class LabelText(object):
    def __init__(self, label_list, ori_path):
        self.label_list = label_list
        self.ori_path = ori_path

    def arrangeLabelText(self, show=True, write=False):
        """
        label+text 未排序
        """
        if write is True:
            write_path = settings.DST_DATA + 'labelText.csv'
            print("new file saved in " + write_path)
            w = open(write_path, 'w')
        content_lines = loading_source(self.ori_path)
        for label, content in zip(self.label_list, content_lines):
            try:
                line = str(label) + '\t' + content.strip()
                if show is True:
                    print(line)
                if write is True:
                    w.write(line)
                    w.write('\n')
            except Exception as e:
                print(e)
                continue
        if write is True:
            w.close()

    def sortByLabel(self, show=True, write=False, algorithm="km"):
        """
        label+text 排序
        """
        if write is True:
            write_path = settings.DST_DATA + algorithm + str(time.time()) + '-sortedLabelText.csv'
            print("new file saved in " + write_path)
            w = open(write_path, 'w')
        content_lines = loading_source(self.ori_path)
        index = np.argsort(self.label_list)
        for i in range(len(index)):
            try:
                line = str(self.label_list[index[i]]) + '\t' + content_lines[index[i]].strip()
                if show is True:
                    print(line)
                if write:
                    w.write(line)
                    w.write('\n')
            except Exception as e:
                print(e)
                continue
        if write is True:
            w.close()
