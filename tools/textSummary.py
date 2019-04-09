# coding=utf-8
import jieba.analyse
import jieba.posseg
import pandas as pd
from settings import *


class TextSummary:
	text = ""
	title = ""
	keywords = list()
	sentences = list()
	summary = list()

	def set_text(self, title, text):
		self.title = title
		self.text = text

	def __split_sentence(self):
		# 通过换行符对文档进行分段
		sections = self.text.split("\n")
		for section in sections:
			if section == "":
				sections.remove(section)

		# 通过分割符对每个段落进行分句
		for i in range(len(sections)):
			# 拿到某一段
			section = sections[i]
			text = ""
			k = 0

			for j in range(len(section)):
				# 取每一个字比较???有点耗时
				char = section[j]
				text = text + char
				if char in ["!",  "。", "？"] or j == len(section)-1:
					text = text.strip()
					sentence = dict()
					sentence["text"] = text
					# 保存句子在段落中的位置, x段, y句
					sentence["pos"] = dict()
					sentence["pos"]["x"] = i
					sentence["pos"]["y"] = k
					# 将处理结果加入self.sentences
					self.sentences.append(sentence)
					text = ""
					k = k + 1

		for sentence in self.sentences:
			sentence["text"] = sentence["text"].strip()
			if sentence["text"] == "":
				self.sentences.remove(sentence)

		# 对文章位置进行标注，通过mark列表，标注出是否是第一段、尾段、第一句、最后一句
		last_pos = dict()
		last_pos["x"] = 0
		last_pos["y"] = 0
		last_pos["mark"] = list()
		for sentence in self.sentences:
			pos = sentence["pos"]
			pos["mark"] = list()
			if pos["x"] == 0:
				pos["mark"].append("FIRSTSECTION")
			if pos["y"] == 0:
				pos["mark"].append("FIRSTSENTENCE")
				last_pos["mark"].append("LASTSENTENCE")
			if pos["x"] == self.sentences[len(self.sentences)-1]["pos"]["x"]:
				pos["mark"].append("LASTSECTION")
			last_pos = pos
		last_pos["mark"].append("LASTSENTENCE")

	def __calc_keywords(self):
		# 载入实体
		jieba.load_userdict(STATIC_DIR + 'dict.txt')
		# 计算tf-idf，取出排名靠前的20个词
		words_best = list()
		words_best = words_best + jieba.analyse.extract_tags(self.text, topK=20)
		# 提取第一段的关键词
		parts = self.text.lstrip().split("\n")
		first_para = ""
		if len(parts) >= 1:
			first_para = parts[0]
		words_best = words_best + jieba.analyse.extract_tags(first_para, topK=5)
		# 提取title中的关键词
		words_best = words_best + jieba.analyse.extract_tags(self.title, topK=3)
		# 将结果合并成一个字符串，并进行分词
		text = ""
		for w in words_best:
			text = text + " " + w
		# 计算词性，提取名词和动词
		words = jieba.posseg.cut(text)
		keywords = list()
		for w in words:
			flag = w.flag
			word = w.word
			if flag.find('n') >= 0 or flag.find('v') >= 0:
				if len(word) > 1:
					keywords.append(word)
		# 保留前20个关键词
		keywords = jieba.analyse.extract_tags(" ".join(keywords), topK=20)
		keywords = list(set(keywords))
		self.keywords = keywords

	def __calc_sentence_weight_by_keywords(self):
		# 计算句子的关键词权重
		for sentence in self.sentences:
			sentence["weightKeywords"] = 0
		for keyword in self.keywords:
			for sentence in self.sentences:
				if sentence["text"].find(keyword) >= 0:
					sentence["weightKeywords"] = sentence["weightKeywords"] + 1

	def __calc_sentence_weight_by_pos(self):
		# 计算句子的位置权重
		for sentence in self.sentences:
			mark = sentence["pos"]["mark"]
			weightPos = 0
			if "FIRSTSECTION" in mark:
				weightPos = weightPos + 2
			if "FIRSTSENTENCE" in mark:
				weightPos = weightPos + 2
			if "LASTSENTENCE" in mark:
				weightPos = weightPos + 1
			if "LASTSECTION" in mark:
				weightPos = weightPos + 1
			sentence["weightPos"] = weightPos

	def __calc_sentence_weight_by_cuewords(self):
		# 计算句子的线索词权重
		index = ["融资", "投融资", "轮", "并入", "收购",
				 "出让", "领投", "募资", "跟投", "增资", "定增", "追加", '教辅', '留学', '教育', '早教', '题库', '单词', '幼教',
				 '课堂','老师', '学习', '教学', '课堂','数千万', '数百万', '备考', '培训', '家教', '知识',
				 '生物', '医疗', '制药', '健康', '保健', '临床', '配方', '实验', '试验', '心理', '精神', '抗体', '医生', '免疫', '肿瘤', '疾病', '生殖',
				 '血糖', '神经', '癌症', '抑郁', '心肺', '血糖', '乳酸', '医学', '再生', '问诊', '诊断', '诊所', '中枢', '基因', '血液', '微创', '生育',
				  '中风', '疼痛', 'health', '胰岛素', '治疗', '患者', '细胞', '靶点', '房颤', '口腔', '护理', '乳腺癌', '疫苗', '矫正', '儿科',
				 '眼科', '医患', '肾病', '肾科', '病历', '医改', '输液', '', '医院']

		content = pd.read_csv(SOURCE_DATA + 'entity.csv')
		index.extend(content.name.to_list())
		for sentence in self.sentences:
			sentence["weightCueWords"] = 0
		for i in index:
			for sentence in self.sentences:
				if sentence["text"].find(str(i)) >= 0:
					sentence["weightCueWords"] += 2
		# index = ["总之", "总而言之", "报导", "新华社", "报道"]
		# for sentence in self.sentences:
		# 	sentence["weightCueWords"] = 0
		# for i in index:
		# 	for sentence in self.sentences:
		# 		if sentence["text"].find(i) >= 0:
		# 			sentence["weightCueWords"] = 1

	def __calc_sentence_weight(self):
		# 句子权重: 句子位置权重 + 2*线索关键词权重 + TF-IDF关键词权重
		self.__calc_sentence_weight_by_pos()
		self.__calc_sentence_weight_by_cuewords()
		self.__calc_sentence_weight_by_keywords()
		for sentence in self.sentences:
			sentence["weight"] = sentence["weightPos"] + 2 * sentence["weightCueWords"] + sentence["weightKeywords"]

	def calc_summary(self, ratio=0.1):
		# 清空变量
		self.keywords = list()
		self.sentences = list()
		self.summary = list()

		# 调用方法，分别计算关键词、分句，计算权重
		self.__calc_keywords()
		self.__split_sentence()
		self.__calc_sentence_weight()

		# 对句子的权重值进行排序
		self.sentences = sorted(self.sentences, key=lambda k: k['weight'], reverse=True)

		# 根据排序结果，取排名占前X%的句子作为摘要
		# print(len(self.sentences))
		for i in range(len(self.sentences)):
			if i < ratio * len(self.sentences):
				sentence = self.sentences[i]
				self.summary.append(sentence["text"])
		return self.summary
