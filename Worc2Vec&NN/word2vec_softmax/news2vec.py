# -*- coding: utf-8 -*-
"""
Created on Sat May 26 14:46:24 2018

@author: Robbery
"""

import numpy as np
import gensim
import os
import re

def getwords(doc):
	#print(doc)
	splitter = re.compile('\\W+')
	# split words and lower
	words = [s.lower() for s in splitter.split(doc) if len(s) >= 2 and len(s) < 20]
	# remove numbers
	words = [word for word in words if word.isalpha()]
	with open('stopwords/stopwords.txt', encoding='utf-8') as f:
		stopwords = f.read()
	stopwords = stopwords.split('\n')
	stopwords = set(stopwords)
	# remove stopwords
	words = [word for word in words if word not in stopwords]
	#print(words)
	return words

def news2vec(news):
    vec = np.zeros((400,))
    num = len(news)
    for word in news:
        try:
            # print(len(np.array(model[word])))
            vec += np.array(model[word])
        except KeyError:
            num -= 1
    if num == 0:
        return 2333         # 对空的新闻返回特殊值
    vec /= num
    return vec
    
# model = gensim.models.KeyedVectors.load_word2vec_format('../corpus.model', binary=True)
# model = gensim.models.KeyedVectors.load_word2vec_format('corpus.model', binary=True)
model = gensim.models.Word2Vec.load('model/corpus.model')
for root, dirs, files in os.walk("data"): #这里可以选择mini或data（完整）
        for file in files:
            realpath = os.path.join(root,file)
            with open(realpath, errors='ignore', encoding='utf-8') as f:
                word_list = eval(f.read())
                file = file[:-4]
                news = []
                vecs = []
                for item in word_list:
                    news.append(getwords(item['title']+'|'+item['content']))
                for new in news:
                    a = news2vec(new)
                    if not isinstance(a,int):       # 不是空的新闻才添加到vec中
                        vecs.append(a.tolist())
                filename = "vec/vec"+file+".txt"
                output = open(filename,'w',encoding = 'utf-8')
                output.write(str(vecs))
                output.close()
