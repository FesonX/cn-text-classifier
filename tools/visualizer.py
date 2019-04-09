import matplotlib.pyplot as plt
from settings import *
import random


def plot_cluster(result, trainingData, numOfClass):
    plt.figure(2)
    # create numOfClass empty lists
    lab = [[] for i in range(numOfClass)]
    index = 0
    for lab_i in result:
        lab[lab_i].append(index)
        index += 1
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^',
                 'g^'] * 3
    for i in range(numOfClass):
        x1 = []
        y1 = []
        for data in trainingData[lab[i]]:
            try:
                x1.append(data[0])
                y1.append(data[1])
            except Exception as e:
                print(e)
        plt.plot(x1, y1, color[i])
    plt.show()


def plot_result(data, cluster_res, cluster_num, algorithm='None'):
    nPoints = len(data)
    scatter_colors = ['blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(cluster_num):
        color = scatter_colors[i % len(scatter_colors)]
        x1 = []
        y1 = []
        for j in range(nPoints):
            if cluster_res[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='o')
        plt.plot(marksize=10)
    plt.savefig(PLOT_DIR + algorithm + '-' + str(random.randint(10, 100)) + str(cluster_num) + '.png')
    plt.show()


def plot_labels(labels: list, training_data):
    unique_labels = set(labels)
    colors = ['blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = [0, 0, 0, 1]

        class_member_mask = (labels == label)
