from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np
import jieba  # 用于中文分词

# 1.例子句子
sentence = "这是一个简单的例子句子。"

# 2.预处理文本（中文分词）
words = list(jieba.cut(sentence))
print("分词结果：", words)

# 3.训练Word2Vec模型
# 通常需要大量语料进行训练，这里简化示例直接用训练好的模型
corpus = [
    list(jieba.cut("这是第一个句子。")),
    list(jieba.cut("这是第二个句子。")),
    list(jieba.cut("这是另一句简单的句子。"))
]

# 训练Word2Vec模型
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# 4.获取词嵌入并编码句子
def encode_sentence(sentence, model):
    words = list(jieba.cut(sentence))
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector

encoded_sentence = encode_sentence(sentence, model)
print("句子编码为向量：", encoded_sentence)
print("向量维度：", encoded_sentence.shape)