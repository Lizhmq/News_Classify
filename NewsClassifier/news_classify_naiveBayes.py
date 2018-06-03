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
		return docprob * catprob
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
	filesize = int(1 * len(data_))
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

newss=["北京时间6月1日，NBA总决赛拉开战幕，勇士通过加时赛在主场以124-114击败骑士，在系列赛总比分上取得1-0的领先。詹姆斯51分，个人季后赛新高。系列赛的第二场比赛，将于下周一（6月4日）继续在勇士主场进行。"
       ,"iPhone 7 Plus在2016年秋天上市，当时在这款产品上苹果就配备了双摄像头。基本上可以说，尽管苹果不是第一个为智能手机配备双摄像头的厂商，但只要苹果开始这么做，广大安卓竞争对手就会开始争相效仿，为自己的产品配备双摄像头，事实证明也的确如此。进入到2018年，基本上所有的旗舰都采用了双摄像头配置，华为的P20 Pro甚至直接上了三摄像头，并且成为了这个世界上拍照效果最好的智能手机之一。考虑到这一点，根据最新的传闻显示，明年的三星Galaxy S10和iPhone X Plus也有可能同样升级到三摄像头配置。"
       ,"功利化、模式化、套路化的作文写作让高考作文的文风疲敝至极，语文的高考命题如何能够自我革新，冲出僵化的应试写作？语文教育终究应该去向何方？多年来穿行在高考、语文教育事业中的漆永祥教授，一边向“高考体”开炮，另一边通过种种方式去寻觅语文教育的更多可能，敲响了改革的鼓声。"
       ,"中新经纬客户端6月2日电(薛宇飞)据不完全统计，截至6月1日，全国至少已经有20个省份公布了2017年城镇非私营单位、私营单位就业人员年平均工资水平，目前，北京分别以131700元和70738元领跑全国。增速方面，多地私营单位就业人员年平均工资增长速度低于非私营单位。"
       ,"据悉，姚晨在与摄影师曹郁结婚后，先后生下一儿一女，一家四口甜蜜幸福。当妈后的姚晨，除了日常工作，把大部分时间用在了照顾孩子上。今年的六一儿童节当天，姚晨就和老公及一对儿女一起外出旅行，享受美好的假期。"
       ,"相比较于旧款，新一代CLS的变化可以说彻头彻尾。前后灯组造型从原来的四边形+柳叶状换成了更为锐利的三角眼和倒梯形状，内部采用的是奔驰矩阵式LED光源，以及L型LED材质的日行灯转向灯集成灯带。除发动机舱盖上的肌肉线条得以保留，整个车身前中后的大部分线条更为圆润。进气格栅采用全新设计的星辉前格栅，搭配大量运动套件，尽显运动气质。尾部采用溜背式设计，圆润的线条从前方延续下来，比例十分协调。后包围两侧的腮孔道气槽不仅能够为新车提升部分空力学性能，还与底部黑色护板嵌入的双边双出镀铬尾排营造运动不错的氛围。"
       ,"火车、公交、景区等场所通常设有儿童票，收费大多是以身高为唯一标准。近年来，随着生活水平的提高，不少孩子5岁时身高就超过了1．2米，仅按照身高作为免票标准有些不合理。六一儿童节即将来临，郑州市动物园联合河南省内15家景区宣布实行新的儿童免票政策。其中，郑州市动物园决定从5月27日起试行1．2米以下或7周岁以下的儿童可免票入园，试行期一个月。"
       ,"据悉，截至2018年3月底，全国医疗卫生机构数达99.3万家，同比增加6715家。其中，医院增加1959家，基层医疗卫生机构增加9541家，专业公共卫生机构减少4654家；公立医院减少373家，民营医院增加2332家；社区卫生服务中心（站）和诊所增加，乡镇卫生院和村卫生室减少；疾病预防控制中心减少27家，卫生监督所（中心）增加14家。"
       ,"搜狐讯 6月2日晚，王力宏上海演唱会第二场如期举行，大哥成龙惊喜现身，不知情的王力宏摔在地上大喊“吓死我了”，大兵小将终于合体，全场简直嗨到爆！成龙在台上对王力宏连说五次我爱你，表示“没办法，你是我的最爱，我一定要过来。你们（歌迷）爱他，我爱他。”两人开心拥抱还大玩亲亲。值得一提的是，当晚王力宏手上拿的龙的传人电吉他是成龙创造与艺术中心为他量身设计的，交情真是不浅呀~"
       ,"iPhone 7 Plus在2016年秋天上市，当时在这款产品上苹果就配备了双摄像头。基本上可以说，尽管苹果不是第一个为智能手机配备双摄像头的厂商，但只要苹果开始这么做，广大安卓竞争对手就会开始争相效仿，为自己的产品配备双摄像头，事实证明也的确如此。进入到2018年，基本上所有的旗舰都采用了双摄像头配置，华为的P20 Pro甚至直接上了三摄像头，并且成为了这个世界上拍照效果最好的智能手机之一。考虑到这一点，根据最新的传闻显示，明年的三星Galaxy S10和iPhone X Plus也有可能同样升级到三摄像头配置。"
       ]

labels=["sports"
        ,"it"
        ,"learning"
        ,"stock"
        ,"yule"
        ,"auto"
        ,"travel"
        ,"health"
        ,"yule"
        ,"it"
        ]

if __name__ == '__main__':
	cl = naivebayes(getwords)
	data = get_dataset()
	train_data, train_target, test_data, test_target = train_and_test_data(data)
	sampletrain(cl, train_data, train_target)
	save_data(cl, 'data')
	# cl = load_data('data')
	predict = []
	# for item in test_data:
	# 	predict.append(cl.classify(item))
	# count = 0
	# for left, right in zip(predict, test_target):
	# 	if left == right:
	# 		count += 1
	# print(count/len(test_target))

else:
	cl = naivebayes(getwords)
	cl = load_data('data')
