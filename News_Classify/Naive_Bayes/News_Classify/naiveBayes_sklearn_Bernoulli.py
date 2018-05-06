import re
import os
import random
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB

def get_dataset():
	data = []
	for root, dirs, files in os.walk('data_text_classification2'):
		for file in files:
			realpath = os.path.join(root, file)
			with open(realpath, errors='ignore', encoding='utf-8') as f:
				words_list = eval(f.read())
				for item in words_list:
					data.append((item['title']+item['content'], item['label']))
	random.shuffle(data)
	return data

def train_and_test_data(data_):
	filesize = int(0.7 * len(data_))
	train_data_ = [each[0] for each in data_[:filesize]]
	train_target_ = [each[1] for each in data_[:filesize]]
	test_data_ = [each[0] for each in data_[filesize:]]
	test_target_ = [each[1] for each in data_[filesize:]]
	return train_data_, train_target_, test_data_, test_target_

data = get_dataset()
train_data, train_target, test_data, test_target = train_and_test_data(data)

nbc = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB(alpha=1.0))])
nbc.fit(train_data, train_target)
predict = nbc.predict(test_data)
count = 0
for left, right in zip(predict, test_target):
	if left == right:
		count += 1
print(count/len(test_target))
'''
if __name__ == '__main__':
	cl = naivebayes(getwords)
	data = get_dataset()
	train_data, train_target, test_data, test_target = train_and_test_data(data)
	sampletrain(cl, train_data, train_target)
	predict = []
	for item in test_data:
		predict.append((cl.classify(item)))
	count = 0
	for left, right in zip(predict, test_target):
		if left == right:
			count += 1
	print(count/len(test_target))
'''
'''
else:
	cl = naivebayes(getwords)
	data = get_dataset()
	train_data, train_target, test_data, test_target = train_and_test_data(data)
	sampletrain(cl, train_data, train_target)
'''