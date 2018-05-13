import sys
import os
import math
import random
import re
import Decision_Tree


def getwords(doc):
	# print(doc)
	splitter = re.compile('\\W+')
	# split words and lower
	words = [s.lower() for s in splitter.split(doc)
	                 if len(s) >= 2 and len(s) < 20]
	# remove numbers
	words = [word for word in words if word.isalpha()]
	with open('stopwords/stopwords.txt', encoding='utf-8') as f:
		stopwords = f.read()
	stopwords = stopwords.split('\n')
	stopwords = set(stopwords)
	# remove stopwords
	words = [word for word in words if word not in stopwords]
	# print(words)
	return words


def get_dataset():
	data = []
	for root, dirs, files in os.walk('data'):
		for file in files:
			realpath = os.path.join(root, file)
			with open(realpath, errors='ignore', encoding='utf-8') as f:
				words_list = eval(f.read())
				for item in words_list:
					data.append(
					    (getwords(item['title'] + '|' + item['content']), item['label']))
	random.shuffle(data)
	data = [i for i in data if i[0]]
	return data


def train_test_data(data_):
	random.shuffle(data_)
	filesize = int(0.7 * len(data_))
	train_data_ = data_[:filesize]
	test_data_ = data_[filesize:]
	return train_data_, test_data_

def train_valid_test_data(data_):
	random.shuffle(data_)
	filesize1 = int(0.6 * len(data_))
	filesize2 = int(0.8 * len(data_))
	train_data_ = data_[:filesize1]
	valid_data_ = data_[filesize1:filesize2]
	test_data_ = data_[filesize2:]
	return train_data_, valid_data_, test_data_

if __name__ == '__main__':
	data = get_dataset()
	train_data, test_data = train_test_data(data)
	train_data1, valid_data1, test_data1 = train_valid_test_data(data)
	root1 = Decision_Tree.GenerateTree(train_data, 1)
	root2 = Decision_Tree.GenerateTree_PrePurn(train_data1, valid_data1, 1)
	root3 = Decision_Tree.GenerateTree(train_data1, 1)
	Decision_Tree.PostPurn(root3, valid_data1)

	pred_true = 0
	for i in test_data:
		if i[1] == Decision_Tree.Predict(root1, i[0]):
			pred_true += 1
	print(pred_true/len(test_data))

	pred_true = 0
	for i in test_data1:
		if i[1] == Decision_Tree.Predict(root2, i[0]):
			pred_true += 1
	print(pred_true/len(test_data1))

	pred_true = 0
	for i in test_data1:
		if i[1] == Decision_Tree.Predict(root3, i[0]):
			pred_true += 1
	print(pred_true/len(test_data1))
