# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:38:05 2018

@author: Robbery
"""

import jieba
import re
import DT
import news_classify_naiveBayes
import NN

rubbish="[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）Ａ-Ｚａ-ｚ０-９＊％＝／＋]+"

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
    
def newsprocess(news):
    news = news.replace(""," ")
    news = re.sub(rubbish," ",news)
    news = '|'.join(jieba.cut(news)).replace("| ","")
    # print(news)
    return news

def Bayes(words):
    return news_classify_naiveBayes.cl.classify(words)

def DecisionTree(words):
    alpha = 150
    beta = 15
    mytree = DT.Tree()
    mytree.load(mytree.root,alpha,beta)
    label = mytree.predict(mytree.root,words)
    return label

def NeuralNetwork(words):
    return NN.predict(words)

newss=["北京时间6月1日，NBA总决赛拉开战幕，勇士通过加时赛在主场以124-114击败骑士，在系列赛总比分上取得1-0的领先。詹姆斯51分，个人季后赛新高。系列赛的第二场比赛，将于下周一（6月4日）继续在勇士主场进行。"
       ,"iPhone 7 Plus在2016年秋天上市，当时在这款产品上苹果就配备了双摄像头。基本上可以说，尽管苹果不是第一个为智能手机配备双摄像头的厂商，但只要苹果开始这么做，广大安卓竞争对手就会开始争相效仿，为自己的产品配备双摄像头，事实证明也的确如此。进入到2018年，基本上所有的旗舰都采用了双摄像头配置，华为的P20 Pro甚至直接上了三摄像头，并且成为了这个世界上拍照效果最好的智能手机之一。考虑到这一点，根据最新的传闻显示，明年的三星Galaxy S10和iPhone X Plus也有可能同样升级到三摄像头配置。"
       ,"功利化、模式化、套路化的作文写作让高考作文的文风疲敝至极，语文的高考命题如何能够自我革新，冲出僵化的应试写作？语文教育终究应该去向何方？多年来穿行在高考、语文教育事业中的漆永祥教授，一边向“高考体”开炮，另一边通过种种方式去寻觅语文教育的更多可能，敲响了改革的鼓声。"
       ,"中新经纬客户端6月2日电(薛宇飞)据不完全统计，截至6月1日，全国至少已经有20个省份公布了2017年城镇非私营单位、私营单位就业人员年平均工资水平，目前，北京分别以131700元和70738元领跑全国。增速方面，多地私营单位就业人员年平均工资增长速度低于非私营单位。"
       ,"据悉，姚晨在与摄影师曹郁结婚后，先后生下一儿一女，一家四口甜蜜幸福。当妈后的姚晨，除了日常工作，把大部分时间用在了照顾孩子上。今年的六一儿童节当天，姚晨就和老公及一对儿女一起外出旅行，享受美好的假期。"
       ,"相比较于旧款，新一代CLS的变化可以说彻头彻尾。前后灯组造型从原来的四边形+柳叶状换成了更为锐利的三角眼和倒梯形状，内部采用的是奔驰矩阵式LED光源，以及L型LED材质的日行灯转向灯集成灯带。除发动机舱盖上的肌肉线条得以保留，整个车身前中后的大部分线条更为圆润。进气格栅采用全新设计的星辉前格栅，搭配大量运动套件，尽显运动气质。尾部采用溜背式设计，圆润的线条从前方延续下来，比例十分协调。后包围两侧的腮孔道气槽不仅能够为新车提升部分空力学性能，还与底部黑色护板嵌入的双边双出镀铬尾排营造运动不错的氛围。"
       ,"火车、公交、景区等场所通常设有儿童票，收费大多是以身高为唯一标准。近年来，随着生活水平的提高，不少孩子5岁时身高就超过了1．2米，仅按照身高作为免票标准有些不合理。六一儿童节即将来临，郑州市动物园联合河南省内15家景区宣布实行新的儿童免票政策。其中，郑州市动物园决定从5月27日起试行1．2米以下或7周岁以下的儿童可免票入园，试行期一个月。"
       ,"据悉，截至2018年3月底，全国医疗卫生机构数达99.3万家，同比增加6715家。其中，医院增加1959家，基层医疗卫生机构增加9541家，专业公共卫生机构减少4654家；公立医院减少373家，民营医院增加2332家；社区卫生服务中心（站）和诊所增加，乡镇卫生院和村卫生室减少；疾病预防控制中心减少27家，卫生监督所（中心）增加14家。"
       ]

#labels=['auto','health','it','learning','sports','stock','travel','yule']
labels=["sports"
        ,"it"
        ,"learning"
        ,"stock"
        ,"yule"
        ,"auto"
        ,"travel"
        ,"health"
        ]

num = len(newss)
print("########## Test ###############")
Bcorrect = 0
Dcorrect = 0
Ncorrect = 0
for i in range(0,num):
    news = newss[i]
    reallabel = labels[i]
    words = news2words(news)
    Blabel = Bayes(newsprocess(news))
    Dlabel = DecisionTree(words)
    Nlabel = NeuralNetwork(words)
    print("News"+str(i)+': '+reallabel)
    print("    Bayes: " + Blabel)
    print("    DT:    " + Dlabel)
    print("    NN:    " + Nlabel)
    if(reallabel == Blabel):
        Bcorrect += 1
    if(reallabel == Dlabel):
        Dcorrect += 1
    if(reallabel == Nlabel):
        Ncorrect += 1
        
print("##########Correctness#############")
print("Bayes: "+str(Bcorrect/num))
print("Decision Tree: "+str(Dcorrect/num))
print("Neural Network: "+str(Ncorrect/num))

    