#! -*- coding: utf-8 -*-

import struct
import os
import six
import codecs
import math
import logging
import platform
import psutil
from humanize import naturalsize
logging.basicConfig(level=logging.INFO, format=u'%(asctime)s - %(levelname)s - %(message)s')


class Progress:
    """显示进度，自己简单封装，比tqdm更可控一些
    iterator: 可迭代的对象；
    period: 显示进度的周期；
    steps: iterator可迭代的总步数，相当于len(iterator)
    """
    def __init__(self, iterator, period=1, steps=None, desc=None):
        self.iterator = iterator
        self.period = period
        if hasattr(iterator, '__len__'):
            self.steps = len(iterator)
        else:
            self.steps = steps
        self.desc = desc
        if self.steps:
            self._format_ = u'%s/%s passed' %('%s', self.steps)
        else:
            self._format_ = u'%s passed'
        if self.desc:
            self._format_ = self.desc + ' - ' + self._format_
        self.logger = logging.getLogger()
    def __iter__(self):
        for i, j in enumerate(self.iterator):
            if (i + 1) % self.period == 0:
                self.logger.info(self._format_ % (i+1))
            yield j


class KenlmNgrams:
    """加载Kenlm的ngram统计结果
    vocab_file: Kenlm统计出来的词(字)表；
    ngram_file: Kenlm统计出来的ngram表；
    order: 统计ngram时设置的n，必须跟ngram_file对应；
    min_count: 自行设置的截断频数。
    """
    def __init__(self, vocab_file, ngram_file, order, min_count):
        self.vocab_file = vocab_file
        self.ngram_file = ngram_file
        self.order = order
        self.min_count = min_count
        self.read_chars()
        self.read_ngrams()
    def read_chars(self):
        f = open(self.vocab_file)
        chars = f.read()
        f.close()
        chars = chars.split('\x00')
        self.chars = [i.decode('utf-8') if six.PY2 else i for i in chars]
    def read_ngrams(self):
        """读取思路参考https://github.com/kpu/kenlm/issues/201
        """
        self.ngrams = [{} for _ in range(self.order)]
        self.total = 0
        size_per_item = self.order * 4 + 8
        def ngrams():
            with open(self.ngram_file, 'rb') as f:
                while True:
                    s = f.read(size_per_item)
                    if len(s) == size_per_item:
                        # https://docs.python.org/3.7/library/struct.html?highlight=unpack#format-characters
                        # count_ngrams 源码：uint64_t count = 0;
                        n = self.unpack('Q', s[-8:])
                        yield s, n
                    else:
                        break
        for s, n in Progress(ngrams(), 100000, desc=u'loading ngrams'):
            if n >= self.min_count:
                self.total += n
                # count_ngrams 源码：typedef unsigned int WordIndex;
                c = [self.unpack('I', s[j*4: (j+1)*4]) for j in range(self.order)]
                kc = []
                # bos(Begin of Sentence)，符号 <s>，count_ngrams 默认以 n 个 <s> 初始填充
                # eos(End of Sentence)，符号 </s>，句子结束时以 </s> 结尾，并重新初始句子
                bos, eos = False, False
                for j in c:
                    if j == 1:
                        bos = True
                    elif j == 2:
                        eos = True
                    elif j > 2:
                        kc.append(self.chars[j])
                kl = len(kc)
                if kl > 0:
                    # 若由甲、乙、丙组成句子，则 count_ngrams 已统计甲、甲乙、甲乙丙组合
                    # 但是没有统计乙丙、丙的组合，所以这里补充此种情况，以此达到对等统计
                    if eos and kl > 1:
                        ks = ['']
                        for j in range(1,kl):
                            ks.append('')
                            for k in range(1,j+1):
                                ks[k] += kc[j]
                                self.ngrams[j-k][ks[k]] = self.ngrams[j-k].get(ks[k], 0) + n
                    if bos:
                        # bos 开始的 count_ngrams 已统计甲、甲乙、甲乙丙组合
                        ks = ''.join(kc)
                        self.ngrams[kl-1][ks] = self.ngrams[kl-1].get(ks, 0) + n
                    else:
                        # 否则补充统计甲、甲乙、甲乙丙组合以及丙、乙丙组合（甲乙丙去重）
                        lks, rks = '', ''
                        for j in range(kl):
                            lks = lks + kc[j]
                            rks = kc[kl-1-j] + rks
                            self.ngrams[j][lks] = self.ngrams[j].get(lks, 0) + n
                            if j < kl-1:
                                self.ngrams[j][rks] = self.ngrams[j].get(rks, 0) + n
    def unpack(self, t, s):
        return struct.unpack(t, s)[0]


def repl(m):
    g1 = m.group(1)
    if g1 == None:
        # 换行符阻止前后字符组词
        g1 = '\n'
    else:
        # 空格将允许前后组词
        g1 += ' '
    return g1


def write_corpus(texts, filename):
    """将语料写到文件中，词与词(字与字)之间用空格隔开
    """
    with codecs.open(filename, 'w', encoding='utf-8') as f:
        for s in Progress(texts, 10000, desc=u'exporting corpus'):
            # s = ' '.join(s) + '\n'
            # 汉字后插一个空格，英文单词（前后为[0-9a-zA-Z_#@\$]组成，中间允许[:/\.&-]字符）不拆分，单词后（可能是空格或汉字）插入空格
            s = re.sub(r'(?:([\u4e00-\u9fa5]|(?:[0-9a-zA-Z_#@\$]+[:/\.&-]+)+[0-9a-zA-Z_#@\$]+|[0-9a-zA-Z_#@\$]+)|[\n:/\.&-]+)[ ]*', repl, s)
            f.write(s)


def count_ngrams(corpus_file, order, vocab_file, ngram_file, memory=0.5):
    """通过os.system调用Kenlm的count_ngrams来统计频数
    其中，memory是占用内存比例，理论上不能超过可用内存比例。
    """
    system_name = platform.system()
    win = system_name.lower() == "windows"
    if win:
        # .\windows\count_ngrams.exe -o 4 --memory=50% --write_vocab_list thucnews.chars <thucnews.corpus >thucnews.ngrams
        cmdline = r'.\windows\count_ngrams.exe'
    else:
        cmdline = './count_ngrams'

    cmdline += ' -o %s --memory=%d%% --write_vocab_list %s <%s >%s'\
        % (order, memory * 100, vocab_file, corpus_file, ngram_file)
    done = os.system(cmdline)
    if done != 0:
        if win and done == -1073740791:
            raise ValueError('count_ngrams.exe 运行可用内存不足，请根据实际情况调低 memory 比例。')
        elif win and done == -1073741515:
            raise ValueError('count_ngrams.exe 运行缺少 boost 等相关必要的 dll 文件。')
        else:
            raise ValueError('Failed to count ngrams by KenLM.')


def filter_ngrams(ngrams, total, min_pmi=1):
    """通过互信息过滤ngrams，只保留“结实”的ngram。
    """
    order = len(ngrams)
    if hasattr(min_pmi, '__iter__'):
        min_pmi = list(min_pmi)
    else:
        min_pmi = [min_pmi] * order
    pl = len(min_pmi) - 1
    output_ngrams = set()
    total = float(total)
    for i in range(order-1, 0, -1):
        ix = min(i, pl)
        for w, v in ngrams[i].items():
            pmi = min([
                total * v / (ngrams[j].get(w[:j+1], total) * ngrams[i-j-1].get(w[j+1:], total))
                for j in range(i)
            ])
            if math.log(pmi) >= min_pmi[ix]:
                output_ngrams.add(w)
    return output_ngrams


class SimpleTrie:
    """通过Trie树结构，来搜索ngrams组成的连续片段
    """
    def __init__(self):
        self.dic = {}
        self.end = True
    def add_word(self, word):
        _ = self.dic
        for c in word:
            if c not in _:
                _[c] = {}
            _ = _[c]
        _[self.end] = word
    def tokenize(self, sent): # 通过最长联接的方式来对句子进行分词
        result = []
        start, end = 0, 1
        for i, c1 in enumerate(sent):
            _ = self.dic
            if i == end:
                result.append(sent[start: end])
                start, end = i, i+1
            for j, c2 in enumerate(sent[i:]):
                if c2 in _:
                    _ = _[c2]
                    if self.end in _:
                        if i + j + 1 > end:
                            end = i + j + 1
                else:
                    break
        result.append(sent[start: end])
        return result


def filter_vocab(candidates, ngrams, order):
    """通过与ngrams对比，排除可能出来的不牢固的词汇(回溯)
    """
    result = {}
    for i, j in candidates.items():
        il = len(i)
        if il < min_len or il > max_len:
            continue
        elif il < 3:
            result[i] = j
        elif il <= order and i in ngrams:
            result[i] = j
        elif il > order:
            flag = True
            for k in range(il + 1 - order):
                if i[k: k+order] not in ngrams:
                    flag = False
                    break
            if flag:
                result[i] = j
    return result


# ======= 算法构建完毕，下面开始执行完整的构建词库流程 =======

import re
import glob

# 语料生成器，并且初步预处理语料
# 这个生成器例子的具体含义不重要，只需要知道它就是逐句地把文本yield出来就行了
def text_generator():
    txts = glob.glob(r'/root/thuctc/THUCNews/*/*.txt')
    for txt in txts:
        d = codecs.open(txt, encoding='utf-8').read()
        d = d.replace(r'\u3000', ' ').strip()
        # 替换类似 <tag> </tag> HTML 标签为换行符，换行符阻止前后字符组词，空格将允许前后组词
        d = re.sub(r'[啊吧啦吗呢呀]+|<\s*[a-zA-Z]+(?:\s+.*|\s*)/?\s*>|<\s*([a-zA-Z]+)\s+.*<\s*/\s*\1\s*>', '\n', d)
        yield re.sub(r'[^\u4e00-\u9fa50-9a-zA-Z _#@\$:/\.&-]+', '\n', d)


min_count = 32 # 最小字词出现频次
min_len = 1    # 最小字词长度
max_len = 12   # 最大字词长度
order = 4
corpus_file = 'thucnews.corpus' # 语料保存的文件名
vocab_file = 'thucnews.chars' # 字符集保存的文件名
ngram_file = 'thucnews.ngrams' # ngram集保存的文件名
output_file = 'thucnews.vocab' # 最后导出的词表文件名
memory = 0.8 # memory 是占用可用内存比例（分母为可用内存），理论上不能超过1

# 获取内存信息
virtual_memory = psutil.virtual_memory()
# 总内存 (以字节为单位)
total_memory = virtual_memory.total
# 可用内存 (以字节为单位)
available_memory = virtual_memory.available
# 根据当前可用内存重新计算 count_ngrams 定义的 memory 参数值（分母为物理内存）
memory = math.floor(10 * (memory * available_memory) / total_memory) / 10
# 打印内存信息
print(f"总内存: {naturalsize(total_memory)}，可用内存: {naturalsize(available_memory)}，重新计算占用物理内存比率：{memory}")


write_corpus(text_generator(), corpus_file) # 将语料转存为文本
count_ngrams(corpus_file, order, vocab_file, ngram_file, memory) # 用Kenlm统计ngram
ngrams = KenlmNgrams(vocab_file, ngram_file, order, min_count) # 加载ngram
ngrams = filter_ngrams(ngrams.ngrams, ngrams.total, [0, 2, 4, 6]) # 过滤ngram
ngtrie = SimpleTrie() # 构建ngram的Trie树

for w in Progress(ngrams, 100000, desc=u'build ngram trie'):
    _ = ngtrie.add_word(w)

candidates = {} # 得到候选词
for art in Progress(text_generator(), 1000, desc='discovering words in articles'):
    # 若原始语料（文章）内容过大并直接使用 tokenize(sent) 函数，
    # 则运行时间主要消耗在后续句子不停切片过程中大内容的内存搬运上。
    for sent in Progress(art.split('\n'), 10000, desc='discovering words in sentences'):
        for w in ngtrie.tokenize(sent): # 预分词
            candidates[w] = candidates.get(w, 0) + 1

# 频数过滤
candidates = {i: j for i, j in candidates.items() if j >= min_count}

# 互信息过滤(回溯)
candidates = filter_vocab(candidates, ngrams, order)

# 输出结果文件
with codecs.open(output_file, 'w', encoding='utf-8') as f:
    for i, j in sorted(candidates.items(), key=lambda s: -s[1]):
        s = '%s %s\n' % (i, j)
        f.write(s)
