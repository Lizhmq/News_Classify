import numpy as np
import gensim
import keras
import re
import jieba

rubbish="[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）Ａ-Ｚａ-ｚ０-９＊％＝／＋]+"
labels=['auto','health','it','learning','sports','stock','travel','yule']

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

def news2words(news):
    news = news.replace(""," ")
    news = re.sub(rubbish," ",news)
    news = '|'.join(jieba.cut(news)).replace("| ","")
    return getwords(news)

def news2vec(news):
    vec = np.zeros((400,))
    num = len(news)
    for word in news:
        try:
            # print(len(np.array(model[word])))
            vec += np.array(model1[word])
        except KeyError:
            num -= 1
    if num == 0:
        raise KeyError         # 对空的新闻返回特殊值
    vec /= num
    return vec

def predict(news):
	vec = news2vec(news)
	vec = [vec]
	# np.reshape(vec, (400, 1))
	# print(vec)
	testX = []
	testX.append(np.array(vec))
	y = model2.predict(testX)
	y = y[0]
	m = 0
	res = 0
	for i in range(8):
		if y[i] > res:
			res = y[i]
			m = i
	return labels[m]

model1 = gensim.models.Word2Vec.load('model/corpus.model')
model2 = keras.models.load_model('model/model_softmax')

# print(predict(news2words(newss[0])))
