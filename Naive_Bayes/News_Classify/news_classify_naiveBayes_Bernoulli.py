import re
import os
import random
import math

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
	return set(words)

class classifier:
	def __init__(self, getfeatures):
		# type-feature counts
		self.fc = {}
		# type-document counts
		self.cc = {}
		# type-fcounts
		self.tf = {}
		# type-totalcounts
		# self.totalfeatures = 0
		self.getfeatures = getfeatures
	def incfc(self, f, cat):
		self.fc.setdefault(cat, {})
		self.fc[cat].setdefault(f, 0)
		self.fc[cat][f] += 1
	def inccc(self, cat):
		self.cc.setdefault(cat, 0)
		self.cc[cat] += 1
	def inctf(self, cat):
		self.tf.setdefault(cat, 0)
		self.tf[cat] += 1
	def inctotalfeature(self):
		self.totalfeatures += 1
	# sum of features of a type
	def typefeature(self, cat):
		# self.fc.setdefault(cat, {})
		# return sum(self.fc[cat].values())
		return self.tf[cat]
	# sum of features
	def totalfeature(self):
		# res = 0
		# for i in self.fc.keys():
		# 	res += sum(self.fc[i].values())
		# return res
		return self.totalfeatures
	# number of feature in a type
	def fcount(self, f, cat):
		if cat in self.fc and f in self.fc[cat]:
			return self.fc[cat][f]
		else:
			return 0.0
	# number of documents in a type 
	def catcount(self, cat):
		if cat in self.cc:
			return self.cc[cat]
		else:
			return 0.0
	# number of documents
	def totalcount(self):
		return sum(self.cc.values())
	# list of types
	def categories(self):
		return self.cc.keys()
	def train(self, item, cat):
		features = self.getfeatures(item)
		mem = {}
		for f in features:
			mem.setdefault(f, 0)
			if mem[f] == 0:
				self.incfc(f, cat)
			mem[f] = 1
			# self.inctotalfeature()
			# self.inctf(cat)
		self.inccc(cat)
	def fprob(self, f, cat):
		if (self.catcount(cat) == 0.0):
			return 0.0
		else:
			return self.fcount(f, cat)/self.typefeature(cat)
	# count of kinds of words in cat 
	def cfcount(self, cat):
		return len(self.fc[cat])
	def weightedprob(self, f, cat, prf, weight=1, ap=0.5):
		basicprob = prf(f, cat)
		totals = sum([self.fcount(f, c) for c in self.categories()])
		bp = ((weight*ap)+(totals*basicprob))/(weight+totals)
		return bp
	def smoothprob(self, f, cat):
		#return (self.fcount(f, cat)+1)/(self.catcount(cat)+len(self.cc))
		#return (self.fcount(f, cat)+1)/(self.catcount(cat)+self.cfcount(cat))
		return (self.fcount(f, cat)+1)/(self.catcount(cat)+2)
class naivebayes(classifier):
	def __init__(self, getfeatures):
		classifier.__init__(self, getfeatures)
	def docprob(self, item, cat):
		features = self.getfeatures(item)
		p = 1;
		for f in features:
			# p *= self.smoothprob(f, cat)
			p += math.log(self.smoothprob(f, cat))
		return p
	def prob(self, item, cat):
		catprob = self.catcount(cat)/self.totalcount()
		docprob = self.docprob(item, cat)
		return docprob + math.log(catprob)
	def classify(self, item):
		max = -1000000000000
		probs = {}
		for cat in self.categories():
			probs[cat] = self.prob(item, cat)
			if probs[cat] > max:
				max = probs[cat]
				best = cat
		return best
	'''
	def predict(self, item):
		if int(random.uniform(1, 100)) % 2 == 0:
			return 'good'
		else:
			return 'bad'
	'''
	
def sampletrain(cl, traindata, traintarget):
    for left, right in zip(traindata, traintarget):
    	cl.train(left, right)
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
'''
cl = classfier(getwords)
sampletrain(cl)
print(cl.fprob('python', 'bad'))
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
else:
	cl = naivebayes(getwords)
	data = get_dataset()
	train_data, train_target, test_data, test_target = train_and_test_data(data)
	sampletrain(cl, train_data, train_target)
'''