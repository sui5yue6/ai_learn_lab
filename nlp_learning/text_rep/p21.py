import pandas as pd
import jieba
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

df = pd.read_csv('data/online_shopping_10_cats.csv', encoding='utf-8')
print(df.head())
print('na',df[df['review'].isna()])
df = df.dropna()

sentences = [[token for token in jieba.lcut(sentence) if token.strip() != ''] for sentence in df['review']]

model = Word2Vec(
    sentences,  # 已分词的句子序列
    vector_size=100,  # 词向量维度
    window=5,  # 上下文窗口大小
    min_count=2,  # 最小词频（低于将被忽略）
    sg=1,  # 1:Skip-Gram，0:CBOW
    workers=4  # 并行训练线程数
)