from dataset import DatasetLoader
from criteo_loader import CriteoLoader
import torch
CACHE_SIZE = 520000000
cache = {}
perfect_hit_cnt = 0
request_cnt = 0
loader = CriteoLoader("/dev/shm/train_processed.txt")
tot_unhit_cnt = 0
tot_req_len = 0
cache_avg = 0

def clean_cache():
    global cache, cache_avg
    if len(cache) < CACHE_SIZE:
        return
    sum = 0
    for each in cache:
        sum += cache[each]
    avg = sum // len(cache) // 2
    to_pop = []
    for each in cache:
        if cache[each] < avg:
            to_pop.append(each)
        cache[each] /= 2
    for each in to_pop:
        cache.pop(each)
    cache_avg = avg
    print("pop cnt %d" % (len(to_pop)))
    
def add_into_cache(keys: list):
    global cache, cache_avg
    remain = CACHE_SIZE - len(cache)
    if remain > 0:
        add_cnt = min(remain, len(keys))
        for i in range(add_cnt):
            if keys[i] not in cache:
                cache[keys[i]] = cache_avg

def simulate_request():
    global request_cnt, perfect_hit_cnt, tot_unhit_cnt, tot_req_len, cache
    request_cnt += 1
    request = [int(each) for each in loader.get(128).reshape(-1)]
    unhit_cnt = sum([(each not in cache) for each in request])
    if unhit_cnt <= int(len(request) * 0.05):
        perfect_hit_cnt += 1
        for each in request:
            cache[each] += 1
    else:
        for each in request:
            if each in cache:
                cache[each] -= 1
    tot_unhit_cnt += unhit_cnt
    tot_req_len += len(request)
    add_into_cache(request)

def print_info():
    global request_cnt, perfect_hit_cnt, tot_unhit_cnt, tot_req_len, cache
    print("perfect hit cnt %d, request cnt %d, perfect hit ratio %f" % (perfect_hit_cnt, request_cnt, perfect_hit_cnt / request_cnt))
    print("tot_unhit_cnt %d, tot_req_len %d" % (tot_unhit_cnt, tot_req_len))
    print("cache len %d" % (len(cache)))
    request_cnt = 0
    perfect_hit_cnt = 0
    tot_unhit_cnt = 0
    tot_req_len = 0

def main():
    while(True):
        for i in range(100):
            simulate_request()
        clean_cache()
        print_info()

if __name__ == "__main__":
    main()