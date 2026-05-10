import jieba
text = "小明毕业于北京大学计算机系"
# 1.精确模式
print(jieba.lcut(text))


# 2.全模式
print(jieba.lcut(text,cut_all=True))

# 3.搜索引擎模式
print(jieba.lcut_for_search(text))

jieba.load_userdict("data/user_dict.txt")
t = jieba.lcut("随着云计算技术的普及，越来越多企业开始采用云原生架构来部署服务，并借助大模型能力提升智能化水平，实现业务流程的自动化与智能决策。")
print(t)
