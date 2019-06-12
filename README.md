# cn-text-classifier
## 中文文本聚类实验
## Chinese Text Cluster Experiments

>实验数据
实验数据来源于多个新闻网站爬取的新闻, 包含教育类510篇, 游戏类231篇, 医疗类388篇, 体育类412篇.
其中, 教育类及医疗类同时是投融资新闻中的细分类别, 用于测试细粒度的聚类能否区分.
>
>有关于新闻内容来源的获取, 请参阅这个仓库: [Finance and Investment Info Spider Collections - 投融资信息爬虫集合
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

>实验内容
1. K-Means 聚类实验

2. Birch 聚类实验

3. DBSCAN 聚类实验

>公众号: 程序员的碎碎念
