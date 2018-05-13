import math
import random


class Node(object):
    '''
    attr是划分属性
    label是当前最多的类别
    attr_down指向子结点
    '''

    def __init__(self, attr_init=None, label_init=None, attr_down_init=[0, 0]):
        self.attr = attr_init
        self.label = label_init
        self.attr_down = attr_down_init


def GenerateTree(train_datas, k):
    new_node = Node(None, None, [0, 0])
    labels = [each[1] for each in train_datas]
    label_count = NodeLabel(labels)
    if label_count:
        new_node.label = max(label_count, key=label_count.get)
    # end if there is only 1 class
    # end if there is no data
        if len(label_count) == 1 or len(labels) == 0:
            return new_node
        if sum(label_count.values()) < 10:
            new_node.attr = None
            return new_node
        new_node.attr = OptAttr_Ent(train_datas)
        data = [0, 0]
        new_node_child = [0, 0]
        data[0], data[1] = divide(train_datas, new_node.attr)
        labels_child = [0, 0]
        new_node.attr_down[0] = GenerateTree(
            data[0], k + 1)
        new_node.attr_down[1] = GenerateTree(
            data[1], k + 1)
    return new_node

def PostPurn(root, valid_datas):
    if root.attr == None:
        return PredictAccuracy(root, valid_datas)
    a1 = 0
    valid_data1, valid_data2 = divide(valid_datas, root.attr)
    l1 = len(valid_data1)
    l2 = len(valid_data2)
    a1_v = PredictAccuracy(root.attr_down[0], valid_data1)
    a2_v = PredictAccuracy(root.attr_down[1], valid_data2)
    a1 = a1_v * l1 / (l1 + l2) + a2_v * l2 / (l1 + l2)

    node = Node(None, root.label, [0, 0])
    a0 = PredictAccuracy(node, valid_datas)

    if a0 > a1 + 0.01:
        root.attr = None
        root.attr_down = [0, 0]
        return a0
    else:
        return PredictAccuracy(root, valid_datas)

def GenerateTree_PrePurn(train_datas, valid_datas, k):
    new_node = Node(None, None, [0, 0])
    labels = [each[1] for each in train_datas]
    label_count = NodeLabel(labels)
    if label_count:
        new_node.label = max(label_count, key=label_count.get)
    # end if there is only 1 class
    # end if there is no data
        if len(label_count) == 1 or len(labels) == 0:
            return new_node
        if sum(label_count.values()) < 10:
            new_node.attr = None
            return new_node
        a0 = PredictAccuracy(new_node, valid_datas)
        new_node.attr = OptAttr_Ent(train_datas)
        data = [0, 0]
        new_node_child = [0, 0]
        data[0], data[1] = divide(train_datas, new_node.attr)
        valid_data1, valid_data2 = divide(valid_datas, new_node.attr)
        labels_child = [0, 0]
        if k > 100:
            for i in range(0, 2):
                new_node_child[i] = Node(None, None, [0, 0])
                labels_child[i] = [each[1] for each in data[i]]
                label_count_child = NodeLabel(labels_child[i])
                new_node_child[i].label = max(
                    label_count_child, key=label_count_child.get)
                new_node.attr_down[i] = new_node_child[i]
            a1 = PredictAccuracy(new_node, valid_datas)
            if a1 > a0:
                new_node.attr_down[0] = GenerateTree_PrePurn(
                    data[0], valid_data1, k + 1)
                new_node.attr_down[1] = GenerateTree_PrePurn(
                    data[1], valid_data2, k + 1)
            else:
                new_node.attr = None
        else:
            new_node.attr_down[0] = GenerateTree_PrePurn(
                data[0], valid_data1, k + 1)
            new_node.attr_down[1] = GenerateTree_PrePurn(
                data[1], valid_data2, k + 1)
    return new_node


def NodeLabel(labels):
    '''
    calculate label and counts
    return a dict
    '''
    label_count = {}
    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    return label_count


def calentropy(x):
    if x > 0:
        return -x * math.log2(x)
    return 0


def InfoEnt(labels):
    ent = 0
    n = len(labels)
    label_count = NodeLabel(labels)
    for key in label_count:
        ent += calentropy(label_count[key] / n)
    return ent


def divide(datas, word):
    data1 = []
    data2 = []
    for i in datas:
        if word in i[0]:
            data2.append(i)
        else:
            data1.append(i)
    return data1, data2


def OptAttr_Ent(datas):
    words = []
    for each in datas:
        words += each[0]
    ent = 10000000
    res = ''
    dic = {}
    for word in words:
        dic.setdefault(word, 0)
        dic[word] += 1
    lst = sorted(zip(dic.values(), dic.keys()), reverse=True)
    for i in range(0, min(100, len(dic))):
        word = lst[i][1]
        data1, data2 = divide(datas, word)
        label1 = [each[1] for each in data1]
        label2 = [each[1] for each in data2]
        count1 = len(label1)
        count2 = len(label2)
        tmp = InfoEnt(label1) * count1 / (count1 + count2) \
            + InfoEnt(label2) * count2 / (count1 + count2)
        if tmp < ent:
            ent = tmp
            res = word
    return res


def Predict(root, sample):
    while root.attr != None:
        if not root.attr in sample:
            root = root.attr_down[0]
        else:
            root = root.attr_down[1]
    return root.label


def PredictAccuracy(root, valid_datas):
    if len(valid_datas) == 0:
        return 0
    pred_true = 0
    for i in valid_datas:
        label = Predict(root, i[0])
        if label == i[1]:
            pred_true += 1
    return pred_true / len(valid_datas)
