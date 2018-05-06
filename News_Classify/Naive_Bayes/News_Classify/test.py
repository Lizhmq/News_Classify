import re

def getwords(doc):
	#print(doc)
	splitter = re.compile('\\W*')
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
	print(words)
	return set(words)

def getwords2(doc):
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
	print(words)
	return set(words)

with open('data_text_classification2/health_test.txt', encoding='utf-8') as f:
	words_list = eval(f.read())
	for item in words_list:
		getwords(item['title']+item['content'])
		getwords2(item['title']+item['content'])
		