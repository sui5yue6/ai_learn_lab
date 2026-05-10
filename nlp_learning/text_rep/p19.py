from gensim.models import KeyedVectors

model_path = 'data/sgns.weibo.word.bz2'
model = KeyedVectors.load_word2vec_format(model_path)

# 1.维数
x = model.vector_size
print(x)


n = len(model.index_to_key)
print(n)

# 3.查看向量
v = model['地铁']
print(v)

# 4.相似度
s = model.similarity('地铁', '图书馆')
print(s)

# 5.最相似
s = model.most_similar(positive=['男人', '女孩'], negative=['男孩'], topn=5)
print(s)
