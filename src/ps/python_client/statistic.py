from dataset import DatasetLoader
import torch

loader = DatasetLoader('/dev/shm/2021/fbgemm_t856_bs65536_0.pt', False, 856, 65536)

sta = {}

def main():
    for i in range(100):
        data = [int(each) for each in loader.get(1)]
        data.sort()
        for i in range(len(data)):
            for j in range(len(data) - i - 1):
                if data[i] not in sta:
                    sta[data[i]] = {}
                if data[j] not in sta[data[i]]:
                    sta[data[i]][data[j]] = 0
                sta[data[i]][data[j]] += 1
        print("---")
    test = []

    for each in sta:
        for key in sta[each]:
            test.append(key)
    
    test.sort()
    print(test)

if __name__ == "__main__":
    main()