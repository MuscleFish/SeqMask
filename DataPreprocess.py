'''
数据预处理模块，读取数据、清理数据、分句、词性变换等
'''
from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from gensim.models.word2vec import Word2Vec
from gensim.models import FastText

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams
import io
import re
import json,simplejson
import copy
import numpy as np

# 数据清理
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", " i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.)\{3\}(?:25[0-5] |2[0-4][0-9]|[01]?[0-9][0-9]?)(/([0-2][0-9]|3[0-2]|[0-9]))?', 'IPv4', text)
    text = re.sub('(CVE\-[0-9]{4}\-[0-9]{4,6})', ' CVE ', text)
    text = re.sub('([a-z][_a-z0-9-.]+@[a-z0-9-]+\.[a-z]+)', ' email ', text)
    text = re.sub('(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', ' IP ', text)
    text = re.sub('([a-f0-9]{32}|[A-F0-9]{32})', ' MD5 ', text)
    text = re.sub('((HKLM|HKCU)\\[\\A-Za-z0-9-_]+)', ' registry ', text)
    text = re.sub('([a-f0-9]{40}|[A-F0-9]{40})', ' SHA1 ', text)
    text = re.sub('([a-f0-9]{64}|[A-F0-9]{64})', ' SHA250 ', text)
    text = re.sub('http(s)?:\\[0-9a-zA-Z_\.\-\\]+.', ' URL ', text)
    text = re.sub('cve-[0-9]{4}-[0-9]{4,6}', ' vulnerability ', text)
    text = re.sub('[a-zA-Z]{1}:\\[0-9a-zA-Z_\.\-\\]+', ' file ', text)
    text = re.sub(r'\b[a-fA-F\d]{32}\b|\b[a-fA-F\d]{40}\b|\b[a-fA-F\d]{64}', ' hash ', text)
    # 去除十六进制
    text = re.sub('\\bx[A-Fa-f0-9]{2}', ' ', text)
    # 缩写
    text = re.sub('\\bno.\d+\\b',' number ',text)

    # y
    text = re.sub('\\bc2\\b','C2',text)
    text = re.sub('\\b[b-z]{1}\\b',' ',text)
    text = re.sub('\\b[a-f0-9]{2}\\b', ' ', text)
    # text = re.sub(r'\b\d+\b',' ',text)
    text = re.sub('\\n', ' ', text)
    # text = re.sub(r'(\.)\1', ' ', text)
    text = re.sub('\d+',' ',text)
    text = re.sub('\'s',' ',text)

    # 去除稀奇古怪的符号
    text = re.sub('[^\w\']', ' ', text)
#     text = re.sub('\W', ' ', text)
    
    text = re.sub("\s+"," ",text)
    text = text.strip(' ')

    return text

def cut_sentences(doc):
    doc = re.sub('\\bno.\d+\\b',' number ',doc)
    doc = re.sub("\s+"," ",doc)
    doc = doc.strip(' ')

    sent_tokenize_list = sent_tokenize(doc)
    sent_tokenize_list = [clean_text(w) for w in sent_tokenize_list]
    
    result = []
    
    for sen in sent_tokenize_list:
        word_list = word_tokenize(sen)
        
        # 删除单词数小于5与大于128的句子
        if len(word_list) < 5 or len(word_list) > 128:
            continue
            
        # 词性还原
        from nltk.stem import WordNetLemmatizer  
        lemmatizer = WordNetLemmatizer()
        
        _sen = []
        for word in word_list:
            word = lemmatizer.lemmatize(word,pos='v') 
            word = lemmatizer.lemmatize(word,pos='n')
            _sen.append(word)
        
        _sen = ' '.join(_sen)
        
        result.append(_sen)
    
    return result

# 读取数据，并处理数据
class DataIo:
    def __init__(self):
        pass

    def read_list(self, sentences):
        return [cut_sentences(s) for s in sentences]
    
    # 读取数据后，对数据进行清理、分句、词性变换，返回文章*句子的二维列表
    def read_csv(self,path,headers,encoding='UTF-8'):
        data = pd.read_csv(path,encoding=encoding)[headers]
        result = []
        shape = data.shape
        row = shape[0]
        col = 1
        if len(shape) == 2:
            col = shape[1]
            
        print('[',end='')
        bar = int(row * col / 10) + 1
        count = 0
        
        for j in range(col):
            for i in range(row):
                result.append(cut_sentences(data.iloc[i,j]))
                if count % bar == 0:
                    print('==',end='')
                count += 1
       
        print('] 数据处理完成')
        
        return result
    
#     def read_pdf(self,path):
#         pass
    
    def read_json(self,path,encoding='UTF-8'):
        result = []
        with open(path,'r',encoding=encoding)as fp:
            data = simplejson.loads(fp.read())
            print('[',end='')
            bar = int(len(data) / 10) + 1
            count = 0
            for val in data:
                try:
                    articles = val['articles']
                    for art in articles:
                        try:
                            content = art['content']
                            result.append(cut_sentences(content))
                        except BaseException:
                            continue
                except BaseException:
                            continue
                        
                if count % bar == 0:
                    print('==',end='')
                count += 1
            print('] 数据处理完成')
        return result

# 词嵌模型
class WordEmbed:
    def __init__(self,model=None,model_type=None):
        self.model = model
        self.type = model_type
        self.size = 1024
        pass
    
    # 训练word2vec
    def word2vec(self,docs=None,corpus_file=None,size=512,alpha=0.025,window=5,min_count=5,max_vocab_size=None,
    sample=0.001,seed=1,workers=8,min_alpha=0.0001,sg=1,hs=0,negative=5,ns_exponent=0.75,cbow_mean=1,
    iter=5,null_word=0,trim_rule=None,sorted_vocab=1,batch_words=100000000,compute_loss=False,callbacks=(),max_final_vocab=None,):
        
        self.size = size
        # 分词
        sentences = []
        for doc in docs:
            for sen in doc:
                sentences.append(word_tokenize(sen))
        
        self.model = Word2Vec(sentences=sentences,corpus_file=corpus_file,size=size,alpha=alpha,window=window,iter=iter,
                              min_count=min_count,max_vocab_size=max_vocab_size,sample=sample,seed=seed,workers=workers
                              ,min_alpha=min_alpha,sg=sg,hs=hs,negative=negative,ns_exponent=ns_exponent,cbow_mean=cbow_mean,
                    null_word=null_word,trim_rule=trim_rule,sorted_vocab=sorted_vocab,batch_words=batch_words,
                              compute_loss=compute_loss,callbacks=callbacks,max_final_vocab=max_final_vocab)
        self.type = 'word2vec'
        
        print('word2vec 词嵌模型训练完成')
    
    # 训练FastText
    def fastText(self,docs,size=512, window=5, min_count=5, epochs = 10):
        self.size = size
        # 分词
        sentences = []
        for doc in docs:
            for sen in doc:
                sentences.append(word_tokenize(sen))
        
        self.model = FastText(size=size,window=window,min_count=min_count)
        self.model.build_vocab(sentences)
        self.model.train(sentences=sentences, total_examples=len(sentences), epochs=epochs)
        self.type = 'fasttext'
        
        print('FastTextModels 词嵌模型训练完成')
    
    # 获取bert预训练模型
    def bert(self,bert_config_path = None,bert_checkpoint_path = None,bert_vocab_path = None):
        self.tokenizer = Tokenizer(bert_vocab_path, do_lower_case=True)  # 建立分词器，忽略大小写
        self.model = build_transformer_model(bert_config_path, bert_checkpoint_path)  # 建立模型，加载权重
        self.type = 'bert'
    
    # 对文章进行embedding
    def embedding(self,docs,max_row=None):
        size = self.size
        result = []
        if self.type == 'word2vec' or self.type == 'fasttext':
            print('[',end='')
            bar = int(len(docs) / 10) + 1
            count = 0
            for doc in docs:
                _doc = []
                for sen in doc:
                    word_list = word_tokenize(sen)
                    for word in word_list:
                        try: 
                            _doc.append(self.model.wv[word])
                        except BaseException:
                            continue

                if max_row:
                    _len = max_row - len(_doc)
                    if _len > 0:
                        for i in range(_len):
                            _doc.append([0]*size)
                    else:
                        _doc = _doc[0:max_row]

                result.append(copy.deepcopy(_doc))
                
                if count % bar ==0:
                    print('==',end='')
                count += 1
                

        elif self.type == 'bert':
            print('[',end='')
            bar = int(len(docs) / 10) + 1
            count = 0
            for doc in docs:
                _doc = []
                for sen in doc:
                    token_ids, segment_ids = self.tokenizer.encode(sen)
                    if len(token_ids) > 512:
                        length = len(token_ids)
                        token_ids = (token_ids[0:500]).append(token_ids[length-1])
                    embedded = self.model.predict([np.array([token_ids]),np.array([segment_ids])])
                    _doc.extend(embedded[0])

                if max_row:
                    _len = max_row - len(_doc)
                    if _len > 0:
                        for i in range(_len):
                            _doc.append([0]*size)
                    else:
                        _doc = _doc[0:max_row]

                result.append(copy.deepcopy(_doc))
                if count % bar == 0:
                    print('==',end='')
                count += 1
        print('] embedding 完成')
        return np.array(result)    
    

