# -*- coding:utf-8 -*-
"""
对mycut处理的数据进一步处理，得到空格分隔的仅保留中文的数据
"""

import time
import numpy as np
import re
from collections import Counter

r = re.compile(u'[a-zA-Z0-9Ａ-Ｚａ-ｚ０-９＊％＝／＋]')

with open('../corpus/corpus_smarty_seg.txt',encoding = 'utf-8') as f:
	texts = f.readlines()

with open('stopwords/stopwords.txt',encoding = 'utf-8') as f:
	stopwords = f.read()
	stopwords = stopwords.split('\n')
	stopwords = set(stopwords)

count = 0
f2 = open('tmp.txt','a',encoding = 'utf-8')
for text in texts:
	text = re.sub(r,'',text)
	text = text.split()
	text = [ts for ts in text if ts not in stopwords]
	for a in text:
		f2.write(a+' ')