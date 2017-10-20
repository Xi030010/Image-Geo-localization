import numpy as np


def get_topd(query_vector, refer_vector, top):

    for j in range(len(query_vector)):
        # cal distance between query img and refer img
        d = np.array([np.sqrt(np.sum(np.square(query_vector[j] - refer))) for refer in refer_vector])
        if j == 0:
            query_top_index = np.argsort(d)
            query_top_d = d[query_top_index]
        else:
            query_top_index = np.concatenate((query_top_index, np.argsort(d)))
            query_top_d = np.concatenate((query_top_d, d[np.argsort(d)]))

    # dimension is topxlen(query_url)
    query_top_index = query_top_index.reshape(top, -1)
    query_top_d = query_top_d.reshape(top, -1)
    return query_top_index, query_top_d


def compare_topd(last_index, last_d, now_index, now_d, top):

    if last_index.shape[0] == 0:
        last_index = now_index
        last_d = now_d
    else:
        last_index = last_index.tolist()
        last_d = last_d.tolist()
        now_index = now_index.tolist()
        now_d = now_d.tolist()

        stackd = []
        stackindex = []
        while len(stackd) == top:
            dd = last_d.pop()
            ddindex = last_index.pop()
            l = len(stackd)
            for i in range(len(now_d)):
               if now_d[i] < dd:
                   stackd.append(now_d[i])
                   stackindex.append(now_index[i])
                   if len(stackd) == top:
                       break
            if len(stackd) == l:
                stackd.append(dd)
                stackindex.append(ddindex)