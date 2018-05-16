import sys
import os
import math
import random
import re
import Decision_Tree
import pp

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

def test_precision(data_, type_):
	# Simple Decision_Tree
	if type_ == 0:
		train_data, test_data = train_test_data(data_)
		root = Decision_Tree.GenerateTree(train_data)
		pred_true = 0
		for i in test_data:
			if i[1] == Decision_Tree.Predict(root, i[0]):
				pred_true += 1
		print('type=0 ', pred_true/len(test_data))
	# PrePurn Decision_Tree
	elif type_ == 1:
		train_data, valid_data, test_data = train_valid_test_data(data_)
		root = Decision_Tree.GenerateTree_PrePurn(train_data, valid_data, 1)
		pred_true = 0
		for i in test_data:
			if i[1] == Decision_Tree.Predict(root, i[0]):
				pred_true += 1
		print('type=1 ', pred_true/len(test_data))
	# PostPurn Decision_True
	else:
		train_data, valid_data, test_data = train_valid_test_data(data_)
		root = Decision_Tree.GenerateTree(train_data)
		Decision_Tree.PostPurn(root, valid_data)
		pred_true = 0
		for i in test_data:
			if i[1] == Decision_Tree.Predict(root, i[0]):
				pred_true += 1
		print('type=2 ', pred_true/len(test_data))


if __name__ == '__main__':
	
	data = get_dataset()
	ppservers=("node-1", "node-2", "node-3")
	job_server = pp.Server(ppservers=ppservers)
	f1 = job_server.submit(test_precision, (data, 0, ), (train_test_data, ), ("Decision_Tree", "re", "math", "random"))
	f2 = job_server.submit(test_precision, (data, 1, ), (train_valid_test_data, ), ("Decision_Tree", "re", "math", "random"))
	f3 = job_server.submit(test_precision, (data, 2, ), (train_valid_test_data, ), ("Decision_Tree", "re", "math", "random"))
	f1()
	f2()
	f3()
	