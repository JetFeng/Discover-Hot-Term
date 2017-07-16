# coding=utf-8
import os
import jieba
import numpy as np
import threadpool
import threading
import copy
stoplist = []


def load_text(filename):
    # format -- (sourceText,[(topic_word, datetime),]*n, n)

    ret_texts = []
    with open(filename, encoding='utf8') as fp:
        lines = fp.readlines()
        for line in lines:
            if line.strip() == '':
                continue

            split_line = line.strip().strip('(').strip(')').split(',')
            if split_line[-1].isdigit():
                nkey = int(split_line[-1])

            ret_text = ','.join([split_line[i] for i in range(len(split_line) - (2 * nkey + 1))])
            ret_texts.append(ret_text)

        return ret_texts


def tokenize(sentence_list):
    with open('stop_words.txt', encoding='utf8') as fp:
        print('正在分词...')

        global stoplist
        stoplist = [w.strip() for w in fp.readlines()]
        stoplist.append(' ')  # 空格也要去掉

        doc_ = [[w.strip() for w in jieba.cut(line) if w not in stoplist]
                for line in sentence_list]

        print('分词完成...')

        # 去除词频为1的词
        from collections import defaultdict
        frequency = defaultdict(int)
        for line in doc_:
            for word in line:
                frequency[word] += 1

        doc_ = [[w for w in line if frequency[w] > 1]
                for line in doc_]


        return doc_


def fetch_corpus(path, update = False):
    # 载入文本集
    import pickle
    docs = []
    seg_list = []

    fname = os.path.split(path)[-1]
    from os.path import join, exists
    if os.path.isdir(path) and os.path.exists(path) and update is True:
        for f in os.listdir(path):
            curpath = join(path, f)
            if os.path.isfile(curpath) and os.path.splitext(curpath)[-1] == '.dat':
                docs_tmp = load_text(curpath)
                docs.extend(docs_tmp)

        docs = list(set(docs))  # 文本简单去重
        docs = docs[:4000]

        # 分词和去停用词
        seg_list = tokenize(docs)

        import pickle
        f_doc = open(fname + '_docs.pkl', 'wb')
        pickle.dump(docs, f_doc)

        f_seg_list = open(fname + '_seg_list.pkl', 'wb')
        pickle.dump(seg_list, f_seg_list)

        return docs, seg_list
    elif not update:

        import pickle
        f_doc = open(fname + '_docs.pkl', 'rb')
        docs = pickle.load(f_doc)

        f_seg_list = open(fname + '_seg_list.pkl', 'rb')
        seg_list = pickle.load(f_seg_list)
        return docs, seg_list

    else:
        raise Exception('目录不存在!')


################################################
#
#  -----------------------
#         w     wh
#  -----------------------
#   t     A     C
#   th    B     D
#  -----------------------
#
#                 (Abelta+B+C+D)(Abelta*D-BC)2
#  x2w,t,belta = ———————————————————————————————
#                (Abelta+B)(C+D)(Abelta+C)(B+D)
#
#

def X2TestTF(t_documents_list, th_documents_list, topK=10):
    import time
    t_start = time.clock()
    from gensim import corpora
    id2word = corpora.Dictionary(t_documents_list)
    x2wt_arr = np.array([0] * len(id2word.token2id), float)
    for w in id2word.token2id.values():
        Abelta = C = B = D = float(0)
        for doc in t_documents_list:
            doc_bow = id2word.doc2bow(doc)  # print(doc_bow) "[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]"
            doc_bow = dict(doc_bow)
            if w in doc_bow.keys():
                count_all = sum(doc_bow.values())
                count_w = doc_bow.get(w, 0)
                TFwj = float(count_w) / count_all
                Abelta += 1
                Abelta += TFwj
            else:
                C += 1
        for doc in th_documents_list:
            if id2word.get(w) in doc:
                B += 1
            else:
                D += 1

        tmp = (Abelta + B + C + D) * (Abelta * D - B * C)**2 / ((Abelta + C) * (Abelta + B) * (B + D) * (C + D))
        x2wt_arr[w] = tmp

    x2wt_arr = np.argsort(-x2wt_arr)
    for i in range(topK):
        print(id2word.get(x2wt_arr[i]))
    t_end = time.clock()
    diff_time = t_end - t_start
    print('%s s' % diff_time)


def X2TestTF_multThread(t_documents_list, th_documents_list, topK=10):
    import time
    t_start = time.clock()
    from gensim import corpora
    id2word = corpora.Dictionary(t_documents_list)
    x2wt_arr = np.array([0] * len(id2word.token2id), float)

    head = 0  # 当前指向第一篇文档
    results_list = []
    mutex_head = threading.Lock()  # 修改head的互斥锁
    mutex_result = threading.Lock()  # 修改results_list的互斥锁

    t_len = len(t_documents_list)
    th_len = len(th_documents_list)
    max_len = t_len if t_len > th_len else th_len

    def _calc(w):
        nonlocal mutex_head, mutex_result, results_list, id2word, head, t_len, th_len, t_documents_list, th_documents_list
        t_doc = []
        th_doc = []
        mutex_head.acquire()
        if head < t_len:
            t_doc = t_documents_list[head]
        if head < th_len:
            th_doc = th_documents_list[head]
        head += 1
        mutex_head.release()

        # print(repr(t_doc))
        # print(repr(th_doc))
        Abeltaj = Bj = Cj = Dj = float(0)
        doc_bow = id2word.doc2bow(t_doc)  # print(doc_bow) "[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]"
        doc_bow = dict(doc_bow)
        if w in doc_bow.keys():
            count_all = sum(doc_bow.values())
            count_w = doc_bow.get(w, 0)
            TFwj = float(count_w) / count_all
            Abeltaj += 1
            Abeltaj += TFwj
        else:
            Cj += 1

        if id2word.get(w) in th_doc:
            Bj += 1
        else:
            Dj += 1

        mutex_result.acquire()
        results_list.append([Abeltaj, Bj, Cj, Dj])
        mutex_result.release()

    for w in id2word.keys():
        Abelta = B = C = D = float(0)

        # if len(t_documents_list) > len(th_documents_list):
        #     th_documents_list.extend([[]] * (len(t_documents_list) - len(th_documents_list)))
        # elif len(t_documents_list) < len(th_documents_list):
        #     t_documents_list.extend([[]] * (len(th_documents_list) - len(t_documents_list)))

        head = 0
        pool = threadpool.ThreadPool(4)
        requests = threadpool.makeRequests(_calc, [w] * max_len)
        [pool.putRequest(req) for req in requests]
        pool.wait()

        result_arr = np.array(results_list)
        Abelta, B, C, D = tuple(np.sum(result_arr, axis=0))
        tmp = (Abelta + B + C + D) * (Abelta * D - B * C)**2 / ((Abelta + C) * (Abelta + B) * (B + D) * (C + D))
        x2wt_arr[w] = tmp
        results_list.clear()

    x2wt_arr = np.argsort(-x2wt_arr)
    for i in range(topK):
        print(id2word.get(x2wt_arr[i]))
    t_end = time.clock()
    diff_time = t_end - t_start
    print('%s s' % diff_time)


# def _calc(w, t_doc, th_doc, id2word):
#     global results_list
#     Abeltaj = Bj = Cj = Dj = float(0)
#     doc_bow = id2word.doc2bow(t_doc)  # print(doc_bow) "[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]"
#     doc_bow = dict(doc_bow)
#     if w in doc_bow.keys():
#         count_all = sum(doc_bow.values())
#         count_w = doc_bow.get(w, 0)
#         TFwj = float(count_w) / count_all
#         Abeltaj += 1
#         Abeltaj += TFwj
#     else:
#         Cj += 1
#
#     if id2word.get(w) in th_doc:
#         Bj += 1
#     else:
#         Dj += 1
#
#     global mutex
#     if mutex.acquire():
#         results_list.append([Abeltaj, Bj, Cj, Dj])
#         mutex.release()


def X2TestTF_multThread2(t_documents_list, th_documents_list, topK=10):
    w_dict = {}
    from gensim import corpora
    id2word = corpora.Dictionary(t_documents_list)

    t_len = len(t_documents_list)
    th_len = len(th_documents_list)

    for t_doc in t_documents_list:
        t_doc_bow = id2word.doc2bow(t_doc)
        t_doc_bow = dict(t_doc_bow)
        for w in t_doc_bow:
            if w_dict.get(w) is None:
                w_dict[w] = []
            TFwj = float(t_doc_bow[w]) / sum(t_doc_bow.values())
            w_dict[w].append(TFwj)

    th_id2word = corpora.Dictionary(th_documents_list)
    x2wt_arr = np.array([0] * len(w_dict), float)
    for w in w_dict.keys():
        print(id2word.get(w))
        Abelta = sum(w_dict[w]) + len(w_dict[w])
        C = t_len - len(w_dict[w])
        term = id2word.get(w)
        th_term_id = th_id2word.token2id.get(term)

        if th_term_id is None:
            B = 0
            D = th_len
        else:
            B = th_id2word.dfs.get(th_term_id)
            D = th_len - B

        x2wt_arr[w] = (Abelta + B + C + D) * (Abelta * D - B * C) ** 2 / ((Abelta + C) * (Abelta + B) * (B + D) * (C + D))

    x2wt_arr = np.argsort(-x2wt_arr)
    # for i in range(topK):
    #     print(id2word.get(x2wt_arr[i]))


def X2Test(t_documents_list, th_documents_list, topK=10):
    from gensim import corpora
    id2word = corpora.Dictionary(t_documents_list)
    x2wt_arr = np.array([0] * len(id2word.token2id), float)
    print(repr(id2word.dfs))
    input()
    for w in id2word.token2id.values():
        A = C = B = D = float(0)
        for doc in t_documents_list:
            doc_bow = id2word.doc2bow(doc)  # print(doc_bow) "[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]"
            doc_bow = dict(doc_bow)
            if w in doc_bow.keys():
                A += 1
            else:
                C += 1
        for doc in th_documents_list:
            if id2word.get(w) in doc:
                B += 1
            else:
                D += 1

        tmp = (A + B + C + D) * (A * D - B * C)**2 / ((A + C) * (A + B) * (B + D) * (C + D))
        x2wt_arr[w] = tmp

    x2wt_arr = np.argsort(-x2wt_arr)
    for i in range(topK):
        print(id2word.get(x2wt_arr[i]))


if __name__ == '__main__':

    # th_doc: t时间之前的文档集合
    # t_doc: t时间段内的文档集合

    th_docs, th_seg_list = fetch_corpus('./input/2', True)
    t_docs,  t_seg_list = fetch_corpus('./input/1', True)

    X2TestTF_multThread2(t_seg_list, th_seg_list)


