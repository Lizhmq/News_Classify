news_classify_naiveBayes.py
自己实现的基于朴素贝叶斯的分类器，准确率约93%-94%

naiveBayes_sklearn_Multinomial.py
调用sklearn中的朴素贝叶斯的库，采用多项式模型，准确率约95%-96%

naiveBayes_sklearn_Bernoulli.py
同上，区别是采取了伯努利模型，准确率也相近




5.5

实现了基本的朴素贝叶斯分类器，原博客采用了weightedprob(对概率加权)来避免可能出现的条件概率为0的问题，个人不是很理解他的内涵，于是采用了被广泛接受的Laplace平滑--smoothprob函数

原博客的docprob函数使用累乘计算概率，考虑到可能出现下溢，改为对数求和--这一步修改后预测准确率有明显提升



5.6

修改了news_classify_naiveBayes.py中smoothprob函数，原来的写法有问题

添加了使用sklearn实现分类器的方法(体验了轮子怎么造之后我们还要会使用别人已经造好的更好的轮子)，模型的训练速度明显提高，准确率也有微小的提升，事实上库中的方法基本与我们实现的相同，推测准确率的提升来源于对数据更好的特征提取、向量化

修改了正则表达式，消除执行的时候的警告："news_classify_naiveBayes.py:10: FutureWarning: split() requires a non-empty pattern match.
  words = [s.lower() for s in splitter.split(doc) if len(s) >= 2 and len(s) < 20]"



5.13
分为多项式模型和伯努利模型
fc[cat][feature]为类cat下单词feature的数目
cc[cat]为类cat的文档数
tf[cat]为类cat下单词总数
totalfeatures为总单词数

fcount返回fc[cat][f]
catcount返回cc[cat]
typefeature返回tf[cat]
totalcount返回总文档数
以上单词重复出现时是记录多次的，适用于多项式模型

在伯努利模型中，对train函数进行一点修改，保证一个文档中相同的单词只被记录一次