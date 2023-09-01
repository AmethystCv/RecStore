import torch, numpy

class CriteoLoader():
    def __init__(self, file_path: str) -> None:
        print("Dataset:", file_path)
        self.load_data(file_path)
        self.offset = 0
        print("Dataset loaded")
    
    def load_data(self, file_path: str):
        self.file_handle = open(file_path, 'r')

    def get(self, batch_size: int):
        data = []
        for i in range(batch_size):
            line = self.file_handle.readline()
            if line == '':
                self.file_handle.seek(0)
                break
            data.append(torch.tensor([int(each) for each in line.split(' ')]))
        return torch.cat(data)

def main():
    loader = CriteoLoader("/home/frw/criteo_script/train_processed2.txt")
    print(loader.get(6))
    print(loader.get(6))
    print(loader.get(5))

if __name__ == "__main__":
    main()