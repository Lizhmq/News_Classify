# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 17:25:39 2018

@author: Robbery
'|'分隔的切词，去除标点和特殊符号
"""
import jieba
import re

def urltolabel(url):
    end = url.index("sohu") - 1
    begin = url[:end].rfind('.') + 1
    if begin == 0:
        begin = 12
    label = url[begin:end]
    if label == "it" and url[begin-5:end] == "club.it":
        label = "club"
    return label
    
inputname='../corpus/news_sohusite_xml_utf_8.dat'
input = open(inputname,'r',encoding = 'utf-8')
lines = input.readlines()
input.close()
News=[]
rubbish="[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）Ａ-Ｚａ-ｚ０-９＊％＝／＋]+"
for i in range(0,len(lines),6):
    news={}
    news["title"] = lines[i+3][14:-16]
    news["content"] = lines[i+4][9:-11]
    news["label"] = urltolabel(lines[i+1])
    News.append(news)

labels = ["it","auto","stock","yule","sports","health","travel","learning"]
results={"it":[],"auto":[],"stock":[],"yule":[],"sports":[],
         "health":[],"travel":[],"learning":[]}

for news in News:
    if news["label"] in labels and len(results[news["label"]]) <= 2000 and news["title"][-4:]!="个人资料" and news["title"] != "创建新论坛" and news["title"][-4:]!="个人介绍":
        title = news["title"].replace(""," ")
        title = re.sub(rubbish," ",title)
        news["title"] = '|'.join(jieba.cut(title)).replace("| ","")
        content = news["content"].replace(""," ")
        content = re.sub(rubbish," ",content)
        news["content"] = '|'.join(jieba.cut(content)).replace("| ","")
        results[news["label"]].append(news)

for label in results.keys():
    filename=label+".txt"
    output = open(filename,'w',encoding = 'utf-8')
    output.write(str(results[label]))
    output.close()
    