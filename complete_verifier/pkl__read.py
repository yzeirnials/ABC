import pickle
import sys
import json


def pkl_to_json(pkl_file, json_file):
    with open(filename, 'rb') as f:
        data = pickle.load(f)


def print_pkl(filename):
    # 读取.pkl文件
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # 使用读取的数据
    print(f"Data Type: {type(data)}")
    print(f"Data Length: {len(data)}")
    # print(f"Data Contens: {data}")
    print(data['summary'])

if __name__ == "__main__":
    print_pkl(sys.argv[1])