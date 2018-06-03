# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 23:53:48 2018

@author: Robbery
"""

import math
import numpy as np
from collections import defaultdict

def calentropy(x):
    if(x>0):
        return -x*math.log2(x)
    return 0

def chooseword(data,alpha):
    d = []
    word_list = []
    word_count = defaultdict(int)
    
    #统计每个词出现的次数
    for key in data.keys():
        d.append(len(data[key]))
        for news in data[key]:
            for word in news:
                word_count[word] += 1
    count_list = []
    for word in word_count.keys():
        count_list.append([word,word_count[word]])
        
    #挑选出现次数最多的alpha个词   
    if(len(count_list) <= alpha):
        for pair in count_list:
            word_list.append(pair[0])
    else:
        count_list = sorted(count_list,key = lambda x:x[1],reverse = True)
        for i in range (0,alpha):
            word_list.append(count_list[i][0])
            
    #特殊情况
    if len(word_list) == 0:
        return 'error'
    if len(word_list) == 1:
        return word_list[0]
    
    #构建矩阵表示含某个词的各类新闻的个数
    matrix = []
    for word in word_list:
        counts=[]
        for key in data.keys():
            count = 0
            for news in data[key]:
                if (word in news):
                    count += 1
            counts.append(count)
        matrix.append(counts)
        
    #计算信息增益，因为原来的信息熵都一样，这里只计算了条件信息熵
    Sum = np.sum(d)
    matrix = np.array(matrix)
    sums = np.sum(matrix,axis = 1)
    sums = np.reshape(sums,(sums.shape[0],1))
    matrix = matrix/sums
    npentropy = np.frompyfunc(calentropy,1,1)
    entropy1 = np.sum(npentropy(matrix).astype(np.float),axis = 1)
    entropy1 = np.reshape(entropy1,(entropy1.shape[0],1))
    sums = sums/Sum
    entropy1 *= sums
    entropy2 = np.sum(npentropy(1-matrix).astype(np.float),axis = 1)
    entropy2 = np.reshape(entropy2,(entropy2.shape[0],1))
    entropy2 *= (1 - sums)
    entropy = entropy1 + entropy2
    
    #对所有词按信息增益排序
    mylist =[]
    for i in range(0,len(word_list)):
        mylist.append([word_list[i],entropy[i][0]])
    mylist = sorted(mylist,key = lambda x:x[1])
    m = mylist[0][1]
    
    #挑选信息增益最大的词
    finallist = []
    for word in mylist:
        if(word[1]>m):
            break
        finallist.append(word[0])
        
    #信息增益最大的词中挑选出现次数最多的词
    finalword = finallist[0]
    count = word_count[finalword]
    for word in finallist:
        if(word_count[word]>count):
            finalword = word
            count = word_count[word]
    return finalword

def divide(data,word):
    left = defaultdict(list)
    right= defaultdict(list)
    for key in data.keys():
        for news in data[key]:
            if(word in news):
                left[key].append(news)
            else:
                right[key].append(news)
    return left,right

class Node(object):
    def __init__(self, word = '', lch = None, rch = None):
        self.word = word
        self.lch = lch
        self.rch = rch

class Tree(object):
    def __init__(self):
        self.root = Node()
        self.mylist = []
        self.count = 0
        
    def build(self, data, curr, alpha, beta):
        if(len(data.keys()) == 1):
            curr.word = list(data.keys())[0]
            return
        s = 0
        for key in data.keys():
            s+=len(data[key])
        #新闻少于beta直接输出种类最多的那类，算是剪枝防止过度拟合
        if(s <= beta):
            keys = list(data.keys())
            word = keys[0]
            count = len(data[word])
            for key in keys:
                if len(data[key])>count:
                    count = len(data[key])
                    word = key
            curr.word = word
            return
        
        word = chooseword(data,alpha)
        if(word =='error'):
            curr.word = 'error'
            return
        left,right = divide(data,word)
        curr.word = word
        curr.lch = Node()
        curr.rch = Node()
        self.build(left,curr.lch,alpha,beta)
        self.build(right,curr.rch,alpha,beta)
        
    def myprint(self,curr):
        print(curr.word)
        if(curr.lch):
            self.myprint(curr.lch)
        if(curr.rch):
            self.myprint(curr.rch)
    
    def predict(self,curr,news):
        if(curr.lch and curr.rch):
            if(curr.word in news):
                return self.predict(curr.lch,news)
            else:
                return self.predict(curr.rch,news)
        else:
            return curr.word
        
    def DFS(self, curr):
        if(curr == None):
            return
        temp = [curr.word]
        if(curr.lch != None):
            temp.append(1)          
        else:
            temp.append(0)
        
        if(curr.rch != None):
            temp.append(1)   
        else:
            temp.append(0)
            
        self.mylist.append(temp)
        self.DFS(curr.lch)
        self.DFS(curr.rch)
        
    def save(self,curr,alpha,beta):
        filename ="DT_alpha_"+str(alpha)+"_beta_"+str(beta)+".txt"
        self.mylist.clear()
        self.DFS(self.root)
        output = open(filename, 'w', encoding='utf-8')
        output.write(str(self.mylist))
        output.close()
     
    def update(self, curr):
        temp = self.mylist[self.count]
        curr.word = temp[0]
        self.count += 1
        if(temp[1] == 1):
            curr.lch = Node()
            self.update(curr.lch)
        if(temp[2] == 1):
            curr.rch = Node()
            self.update(curr.rch)
        
    def load(self,curr,alpha,beta):
        filename ="DT_alpha_"+str(alpha)+"_beta_"+str(beta)+".txt"
        f = open(filename, 'r', encoding='utf-8')
        self.mylist = eval(f.read())
        f.close()
        self.count = 0
        self.update(self.root)

"""
alpha = 100
beta = 10
mytree = Tree()
mytree.load(mytree.root,alpha,beta)
mytree.save(mytree.root,alpha*2,beta)
"""

        