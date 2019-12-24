# 速度更快、效果更好的中文新词发现

复现了之前的<a href="https://kexue.fm/archives/4256">《【中文分词系列】 8. 更好的新词发现算法》</a>中的新词发现算法。

- 算法细节： https://kexue.fm/archives/4256
- 复现细节： https://kexue.fm/archives/6920

## 实测

在经过充分训练的情况下，用bakeoff2005的pku语料进行测试，能得到0.765的F1，优于ICLR 2019的<a href="https://openreview.net/forum?id=r1NDBsAqY7" target="_blank">《Unsupervised Word Discovery with Segmental Neural Language Models》</a>的0.731

（注：这里是为了给效果提供一个直观感知，比较可能是不公平的，因为我不确定这篇论文中的训练集用了哪些语料。但我感觉在相同时间内本文算法会优于论文的算法，因为直觉论文的算法训练起来会很慢。作者也没有开源，所以有不少不确定之处，如有错谬，请读者指正。）

## 使用

使用前务必通过
```
chmod +x count_ngrams
```
赋予`count_ngrams`可执行权限，然后修改`word_discovery.py`适配自己的数据，最后执行
```
python word_discovery.py
```

## 更新
- 2019.12.04: 兼容python3，在python2.7和python3.5下测试通过。

## 交流
QQ交流群：67729435，微信群请加机器人微信号spaces_ac_cn
