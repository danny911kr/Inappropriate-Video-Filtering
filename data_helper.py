# -*- coding: utf-8 -*-:
"""
데이터 가공 클래스
Data : class
def decompose_as_one_hot : 하나의 character를 ord()를 통해 ascii 코드로 변환(0~127 사이의 번호로)
                        추후에 0이라는 숫자는 masking이라는 작업을 통해 keras가 무시하게 만들거기 때문에
                        0이라는 숫자가 데이터로 표현되면 안되므로, 이 함수에서 ord()를 통해 나온 숫자에 +1을 해서
                        원소가 한개인 리스트 형태로 결과값을 도출한다.

def character_to_one_hot : string 형태의 sentence를 받아 sentence를 이루고 있는 character 하나하나에 대해
                        decompose_as_one_hot 함수를 적용하여 빈 리스트에 extend 메소드를 사용하여 결합시킨다.
                        그러면 예를 들어 숫자가 정확하진 않지만, hello -> [42,62,53,53,58] 처럼 나오게 된다.
                        
def preprocess : character_to_one_hot을 통해 하나의 sentence를 벡터의 리스트로 표현하면,
                예를 들어 hello는 다섯개의 벡터로 이루어진 리스트일 것이다.
                이때 만약 string의 최대 길이가 150이라면, (1,150) 형태의 행렬의 맨 앞부터 다섯개의 벡터를 순서대로 채우고
                나머지 뒷부분엔 모두 0으로 채운다. 이를 zero-padding이라 한다.

def load_data_and_labels : mini-batch가 30이라 가정하면, 30개의 문장이 list형태로 이 함수안에 들어가
                            각 문장에 대해 preprocess를 진행하여 나온 (1,string 최대길이) 형태의 행렬들을
                            모두 모아 하나의 리스트로 표현한다. 그러면 (30, string 최대길이) 형태의 행렬이 나올 것이다.
                            
def _batch_loader : 데이터를 batch size만큼 잘라서 전달하는 함수
"""

import numpy as np
import codecs
import re
import itertools
from collections import Counter 
from csv import DictReader
from csv import DictWriter

class Data:
    def __init__(self, file_instances):
        # Load data
        self.instances = self.read(file_instances)
        self.sentences = {} #문장
        for instance in self.instances:
            instance['seqid'] = int(instance['seqid']) #각 인스턴스 별로 번호 지정
        
        for sentence in self.instances:
            self.sentences[sentence['seqid']] = sentence['sentence'] #각 인스턴스 번호를 인덱스로 sentences dictionary set 만든다.

    #csv 문서 읽어오는 함수          
    def read(self, filename):
        rows = []
        # file 열기
        with open(filename, "r") as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)
        return rows
    
        
#각 character를 one-hot-encoding으로 표현하기 위한 전처리 과정이다.
#ascii코드표에 의거, 특수문자, 숫자, 영어 모두 포함하면 0~127 즉 128차원으로 모든 character들을 표현 가능하다.
#밑에 print 해보면 어떻게 나오는지 확인 가능하다.
def decompose_as_one_hot(in_char, warning=True):
    if in_char < 128:  # ASCII CODE표 참조!!
        char_uni = in_char+1
        return [char_uni]  # ex) in_char = 'a' -> [char_uni] = [33]
    else:
        if warning: # 128보다 클때, 예를 들어 한글이라던가 다른 단어
            print('Unhandled character:', chr(in_char), in_char)
        return []
        
def character_to_onehot(string, warning=True):
    tmp_list = []
    for x in string:
        tmp_list.extend(decompose_as_one_hot(ord(x)))
    return tmp_list #ex) string = 'abcde' -> tmp_list = [33,34,35,36,37]
#print(character_to_onehot("abcdefg안녕안녕!@#$ d d d")) #테스트 해볼 것!
print(character_to_onehot("shin woong bi"))
def preprocess(data, max_length):
    vectorized_data = [character_to_onehot(data)]
    # data = 'abcde' -> vectorized_data = [33,34,35,36,37]
    zero_padding = np.zeros((1, max_length), dtype=np.int32)
    # zero_padding = [0,0,0,0,0,0,0,0,0,.....] : (1,maxstringlength) 형태
    for idx, seq in enumerate(vectorized_data):
        length = len(seq)
        if length >= max_length:  #vectorized_data 길이가 maxstringlength보다 크다면
            length = max_length
            #zero_padding = [33,34,35,36,37,38,39,40.....] : (1,maxstringlength) 형태
            zero_padding[idx, :length] = np.array(seq)[:length]  
        else:
            #zero_padding = [33,34,35,36,37,0,0,0,0,0,0.....] : (1,maxstringlength) 형태
            zero_padding[idx, :length] = np.array(seq)

    return zero_padding

def load_data_and_labels(file_instances,max_length):
    
    # Load data from files
    train_sentences = []
    train_sentimentlabels = []
    data = Data(file_instances)
    
    for instance in data.instances:
        sentence_id = instance['seqid']
        sentiment_label = instance['sentiment_label']
        train_sentences.extend(preprocess(data.sentences[sentence_id], max_length))
        train_sentimentlabels.append((sentence_id,sentiment_label))

    sentiment_results = np.zeros((len(train_sentimentlabels),3)) #긍정 부정 중립
    
    for i, sentiment_label in train_sentimentlabels:
        if sentiment_label == '2':     #2면 부정
            sentiment_results[i,2] = 1 #[0,0,1]
        elif sentiment_label == '1':   #1면 긍정
            sentiment_results[i,1] = 1 #[0,1,0]
        else:                          #0면 중립
            sentiment_results[i,0] = 1 #[1,0,0]

    return train_sentences, sentiment_results

def _batch_loader(iterable, n=1):
    length = len(iterable) #list 형태의 전체 데이터
    for n_idx in range(0, length, n):          
        yield iterable[n_idx:min(n_idx + n, length)]