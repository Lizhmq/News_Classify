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
	def __init__(self, getfeatures, fc={}, cc={}):
		# feature counts
		self.fc = fc
		# document counts
		self.cc = cc
		self.getfeatures = getfeatures
	def incf(self, f, cat):
		self.fc.setdefault(f, {})
		self.fc[f].setdefault(cat, 0)
		self.fc[f][cat] += 1
	def incc(self, cat):
		self.cc.setdefault(cat, 0)
		self.cc[cat] += 1
	def fcount(self, f, cat):
		if f in self.fc and cat in self.fc[f]:
			return self.fc[f][cat]
		else:
			return 0.0
	def catcount(self, cat):
		if cat in self.cc:
			return self.cc[cat]
		else:
			return 0.0
	# count of documents
	def totalcount(self):
		return sum(self.cc.values())
	# count of types
	def categories(self):
		return self.cc.keys()
	def train(self, item, cat):
		features = self.getfeatures(item)
		for f in features:
			self.incf(f, cat)
		self.incc(cat)
	def fprob(self, f, cat):
		if (self.catcount(cat) == 0.0):
			return 0.0
		else:
			return self.fcount(f, cat)/self.catcount(cat)
	def weightedprob(self, f, cat, prf, weight=1, ap=0.5):
		basicprob = prf(f, cat)
		totals = sum([self.fcount(f, c) for c in self.categories()])
		bp = ((weight*ap)+(totals*basicprob))/(weight+totals)
		return bp
	def smoothprob(self, f, cat):
		#return (self.fcount(f, cat)+1)/(self.catcount(cat)+len(self.cc))
		return (self.fcount(f, cat)+1)/(self.catcount(cat)+self.totalcount())
class naivebayes(classifier):
	def __init__(self, getfeatures, fc={}, cc={}):
		classifier.__init__(self, getfeatures, fc, cc)
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

def save_data(cl, path):
	with open(path, 'w') as f:
		f.write(str(cl.fc)+'\n'+str(cl.cc)+'\n')
	f.close()

def load_data(path):
	with open(path) as f:
		fc = eval(f.readline())
		# f.readline()
		cc = eval(f.readline())
	cl = naivebayes(getwords, fc=fc, cc=cc)
	return cl

if __name__ == '__main__':
	cl = naivebayes(getwords)
	data = get_dataset()
	train_data, train_target, test_data, test_target = train_and_test_data(data)
	sampletrain(cl, train_data, train_target)
	save_data(cl, 'data')
	# cl = load_data('data')
	for i in range(5):
		print(test_data[i])
	predict = []
	for item in test_data:
		predict.append((cl.classify(item)))
	count = 0
	for left, right in zip(predict, test_target):
		if left == right:
			count += 1
	print(count/len(test_target))

else:
	cl = naivebayes(getwords)
	cl = load_data('data')
