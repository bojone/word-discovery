#! -*- coding: utf-8 -*-

import os
import jieba
jieba.set_dictionary('thucnews.vocab') # 手动截取前50000个词


jieba.lcut(u'今天天气很不错')


F = open('myresult.txt', 'w')

with open('../testing/pku_test.txt') as f:
    for l in f:
        l = l.decode('gbk').strip()
        l = ' '.join(jieba.cut(l, HMM=False))
        l += '\r\n'
        l = l.encode('gbk')
        F.write(l)


F.close()

os.system('./score ../gold/pku_training_words.txt ../gold/pku_test_gold.txt myresult.txt > score.txt')
os.system('cat score.txt')
