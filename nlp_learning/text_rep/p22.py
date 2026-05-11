import pandas as pd
import jieba
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


v= KeyedVectors.load_word2vec_format('data/word2vec.txt')['房间']
print(v)
