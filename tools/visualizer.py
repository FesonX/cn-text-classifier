import matplotlib.pyplot as plt


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

    # draw central point
    # x1 = []
    # y1 = []
    # for point in clf.cluster_centers_:
    #     try:
    #         x1.append(point[0])
    #         y1.append(point[1])
    #     except Exception:
    #         pass
    # plt.plot(x1, y1, 'rv')
    plt.show()


def plot_result(data, cluster_res, cluster_num):
    nPoints = len(data)
    scatter_colors = [ 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(cluster_num):
        color = scatter_colors[i % len(scatter_colors)]
        x1 = []
        y1 = []
        for j in range(nPoints):
            if cluster_res[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')