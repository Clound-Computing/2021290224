import json
import os
import thulac
import pickle as pkl
import numpy as np
import pickle
from gci_framework import GCI
import pandas as pd
from sklearn.metrics import classification_report

if not os.path.exists('data/preprocessed_data.pkl'):
    thu1 = thulac.thulac(seg_only=True)
    fact = []
    accu_clean = []

    with open('data/data.json', 'r', encoding='utf-8') as f:
        for line in f:
            cur_accu = json.loads(line)["accusation"]
            fact.append(json.loads(line)["fact"])
            accu_clean.append(cur_accu)

    fact_split = []
    for line in fact:
        words = thu1.cut(line, text=True)
        fact_split.append(words.split(' '))

    def load_stop_words(stop_word_file):
        stop_words = []
        for line in open(stop_word_file, 'r', encoding='gbk'):
            if line.strip()[0:1] != "#":
                for word in line.split():
                    stop_words.append(word)
        return stop_words

    def havenumber(s):
        f = False
        for i in s:
            if (i >= '0') and (i <= '9'):
                f = True
                break
        return f

    stop_word_file = 'data/ZHstopwords.txt'
    stop_words = load_stop_words(stop_word_file)
    fact_clean = []
    for i in fact_split:
        cur_clean = []
        for j in i:
            if (not j in stop_words) and (not '某' in j) and not havenumber(j):
                cur_clean.append(j)
        fact_clean.append(cur_clean)

    data = {
        'fact_original': fact,
        'fact_clean': fact_clean,
        'accu_clean': accu_clean
    }
    with open('data/preprocessed_data.pkl', 'wb') as f:
        pkl.dump(data, f)
else:
    with open('data/preprocessed_data.pkl', 'rb') as f:
        data = pkl.load(f)
print('Done preparing CAIL data')

if not os.path.exists('data/used_wv.pkl'):
    from gensim.models import KeyedVectors

    wv_from_text = KeyedVectors.load_word2vec_format(
        'data/Tencent_AILab_ChineseEmbedding.txt', binary=False)
    print('Done loading word embedding')

    used_wv = {}
    oov = set()
    exact_oov = 0
    for i in data['fact_clean']:
        for j in i:
            if not j in used_wv:
                if j in wv_from_text.vocab:
                    used_wv[j] = wv_from_text.word_vec(j).tolist()
                else:
                    oov.add(j)
                    ebd = []
                    for k in j:
                        if k in wv_from_text.vocab:
                            ebd.append(wv_from_text.word_vec(k).tolist())
                    if len(ebd) > 0:
                        used_wv[j] = np.mean(np.array(ebd), axis=0)
                    else:
                        used_wv[j] = np.random.rand(200) * 2 - 1
                        exact_oov += 1

    print(len(used_wv), len(oov), exact_oov)
    with open('data/used_wv.pkl', 'wb') as f:
        pkl.dump(used_wv, f)
else:
    with open('data/used_wv.pkl', 'rb') as f:
        used_wv = pkl.load(f)
print('Done preparing word vectors')

if not os.path.exists('data/vocab.pkl'):
    MAX_VOCAB_SIZE = 10000
    UNK, PAD = '<UNK>', '<PAD>'

    def build_vocab(text, max_size, min_freq):
        vocab_dic = {}
        for line in text:
            for word in line:
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[
            :max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
        return vocab_dic

    word_to_id = build_vocab(data['fact_clean'], max_size=MAX_VOCAB_SIZE, min_freq=1)
    pkl.dump(word_to_id, open('data/vocab.pkl', 'wb'))

    embeddings = np.random.rand(len(word_to_id), 200)
    with open('data/used_wv.pkl', 'rb') as f:
        wv = pkl.load(f)
    for i in wv:
        if i in word_to_id:
            idx = word_to_id[i]
            embeddings[idx] = wv[i]
    np.savez_compressed('data/embeddings.npz', embeddings=embeddings)
    print('Done generating vocab and embeddings')




#处理假新闻检测数据部分

# # 加载数据
# train_df = pd.read_csv('data/train_data.csv')
# test_df = pd.read_csv('data/test_data.csv')
# gci = GCI()
# # 构建因果图
# causal_graph = gci.build_causal_graph(train_df['original_text'])
# with open('data/causal_graph.pkl', 'wb') as f:
#     pickle.dump(causal_graph, f)
#
# # 估计因果强度
# causal_strength = gci.estimate_causal_strength(causal_graph)
# with open('data/causal_strength.pkl', 'wb') as f:
#     pickle.dump(causal_strength, f)
#
# # 训练GCI模型
# gci.train_model(causal_graph, causal_strength)
# predictions = gci.predict(test_df['original_text'])
# test_df['predictions'] = predictions
# test_df.to_csv('data/predictions.csv', index=False)
#
# # 评估模型性能
# report = classification_report(test_df['label'], test_df['predictions'])
# print(report)
#
# # 保存评估报告
# with open('data/classification_report.txt', 'w') as f:
#     f.write(report)











