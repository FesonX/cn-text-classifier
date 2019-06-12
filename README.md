# cn-text-classifier
## 中文文本聚类实验
## Chinese Text Cluster Experiments

>实验数据

实验数据来源于多个新闻网站爬取的新闻, 包含教育类510篇, 游戏类231篇, 医疗类388篇, 体育类412篇.
其中, 教育类及医疗类同时是投融资新闻中的细分类别, 用于测试细粒度的聚类能否区分.

**有关于新闻内容来源的获取, 请参阅这个仓库:** [Finance and Investment Info Spider Collections - 投融资信息爬虫集合
](https://github.com/FesonX/finvest-spider)


>实验步骤

文本聚类的一般步骤是:
1. 文本预处理
包含分段分句, 分词及去停用词等
2. 语料向量化或词袋化
本实验使用了 `sklearn` 的 TF-IDF 相关包
3. 文本降维
常见的降维有 PCA 主成分降维, TSVD 截断奇异值分解降维, t-SNE降维等, 本实验使用 PCA 降维, t-SNE 更适用于图像和视频等降维, 速度较慢.
4. 应用聚类算法并调参
5. (可选)结果可视化及聚类效果评判
聚类结果可视化已在 `tools` 文件夹中的 `visualizer.py` 中实现, 鉴于 DBSCAN 有识别噪声的能力, 在该实验中单独加入噪声可视化.
聚类效果评判分为外部信息指标和内部信息指标, 外部信息指标依靠标注好的数据 `src/labeled_data.csv`, 相关知识请参阅:
[无监督学习 - 聚类度量指标](https://www.jianshu.com/p/611ecd46bd35)

>实验结果预览

**K-Means 聚类实验**
![K-Means](https://upload-images.jianshu.io/upload_images/5530017-81f526af29d27a13.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```shell
------K-Means Experiment-------
adjusted_rand_score: 0.993424
FMI: 0.993424
Silhouette: 0.392882
 CHI: 610.273556
------End------
```

**Birch 聚类实验**
![Birch](https://upload-images.jianshu.io/upload_images/5530017-fd9b85232307e60e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```shell
-------Birch Experiment-------
adjusted_rand_score: 0.978233
FMI: 0.978233
Silhouette: 0.392189
 CHI: 605.710339
------End------
```

**DBSCAN 聚类实验**
![DBSCAN](https://upload-images.jianshu.io/upload_images/5530017-7673094ee2fb30d0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
```shell
------DBSCAN Experiment-------
adjusted_rand_score: 0.905969
FMI: 0.905969
Silhouette: 0.379187
CHI: 366.856356
Estimated number of noise points: 102 
------End------
```

>公众号: 程序员的碎碎念

