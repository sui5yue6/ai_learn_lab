import pandas as pd
import jieba
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

df = pd.read_csv('data/online_shopping_10_cats.csv', encoding='utf-8')
print(df.head())
print('na', df[df['review'].isna()])
df = df.dropna()
# 这样会有空格
# 因此需要对 jieba.lcut(sentence)进行多一层遍历 if语句是进行过滤的 空格，以及\t这些没有意义的
# sentences = [jieba.lcut(sentence) for sentence in df['review']]

sentences = [[token for token in jieba.lcut(sentence) if token.strip() != ''] for sentence in df['review']]

model = Word2Vec(
    sentences,  # 已分词的句子序列
    vector_size=100,  # 词向量维度
    window=5,  # 上下文窗口大小
    min_count=2,  # 最小词频（低于将被忽略）
    sg=1,  # 1:Skip-Gram，0:CBOW
    workers=10  # 并行训练线程数
)
model.wv.save_word2vec_format('data/word2vec.txt')

# 如果训练的有空格这些的，是会报错的
v= KeyedVectors.load_word2vec_format('data/word2vec.txt')['房间']
print(v)
