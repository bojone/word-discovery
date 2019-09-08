# 速度更快、效果更好的中文新词发现

复现了之前的<a href="https://kexue.fm/archives/4256">《【中文分词系列】 8. 更好的新词发现算法》</a>中的新词发现算法。

算法细节看： https://kexue.fm/archives/4256
复现细节看： https://kexue.fm/archives/6920

实测结果是：在经过充分训练的情况下，用bakeoff2005的pku语料进行测试，能得到0.746的F1，优于ICLR 2019的<a href="https://openreview.net/forum?id=r1NDBsAqY7" target="_blank">《Unsupervised Word Discovery with Segmental Neural Language Models》</a>的0.731
