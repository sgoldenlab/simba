import redis
import json
import numpy as np
import multiprocessing


data = np.random.randint(0, 100, (1000, 5))
r = redis.Redis(host='localhost', port=6379)
data_split =np.array_split(data, 5)
for i in range(len(data_split)):
    arr_json = json.dumps(data_split[i].tolist())
    r.set(i, arr_json)

def test_func(val):
    arr_loaded = np.array(json.loads(r.get(str(i))))
    print(arr_loaded.shape)


chunk_lst = [x for x in range(5)]
with multiprocessing.Pool(7, maxtasksperchild=1) as pool:
    for cnt, result in enumerate(pool.imap(test_func, chunk_lst, chunksize=1)):
        pass



