import numpy as np
import json
import random
from janome.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize

def numpy_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist() # 将NumPy数组转换为列表
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def json_dump_list(data:list, file_path:str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False,default=numpy_to_json)


train_data = []
valid_data = []
test_data = []

vocalbulary_en = {}
vocalbulary_jp = {}

vocalbulary_en_list = []
vocalbulary_jp_list = []

t_jp = Tokenizer()

vocalbulary_en_list.append("<unk>")
vocalbulary_en_list.append("<PAD>")
vocalbulary_en_list.append("<SOS>")
vocalbulary_en_list.append("<EOS>")
vocalbulary_jp_list.append("<unk>")
vocalbulary_jp_list.append("<PAD>")
vocalbulary_jp_list.append("<SOS>")
vocalbulary_jp_list.append("<EOS>")


vocalbulary_en["<PAD>"] = 0
vocalbulary_en["<SOS>"] = 1
vocalbulary_en['<EOS>'] = 2
vocalbulary_en['<unk>'] = 3
vocalbulary_jp["<PAD>"] = 0
vocalbulary_jp["<SOS>"] = 1
vocalbulary_jp["<EOS>"] = 2
vocalbulary_jp["<unk>"] = 3

with open('./data/eng_jpn.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    text_jp, text_en = line.rsplit('\t', 1)

    tokens_en = word_tokenize(text_en)
    tokens_jp = [token.surface for token in t_jp.tokenize(text_jp)]

    temp_dict = {"text_en": text_en, "text_jp": text_jp, "tokens_en": tokens_en, "tokens_jp": tokens_jp}



    ##随机划分数据集 8/1/1 train/valid/test
    rand_int = random.randint(1, 10)

    if  rand_int <= 8 :
        for token in tokens_en:
            if token not in vocalbulary_en:
                vocalbulary_en[token] = len(vocalbulary_en)
                vocalbulary_en_list.append(token)
        for token in tokens_jp:
            if token not in vocalbulary_jp:
                vocalbulary_jp[token] = len(vocalbulary_jp)
                vocalbulary_jp_list.append(token)

        train_data.append(temp_dict)
    elif rand_int == 9:
        valid_data.append(temp_dict)
    else:
        test_data.append(temp_dict)

with open('./data/train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False)

with open('./data/valid.json', 'w', encoding='utf-8') as f:
    json.dump(valid_data, f, ensure_ascii=False)

with open('./data/test.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False)

with open('./data/vocalbulary_en.json', 'w', encoding='utf-8') as f:
    json.dump(vocalbulary_en, f, ensure_ascii=False)

with open('./data/vocalbulary_jp.json', 'w', encoding='utf-8') as f:
    json.dump(vocalbulary_jp, f, ensure_ascii=False)

with open('./data/vocalbulary_en_list.json', 'w', encoding='utf-8') as f:
    json.dump(vocalbulary_en_list, f, ensure_ascii=False)

with open('./data/vocalbulary_jp_list.json', 'w', encoding='utf-8') as f:
    json.dump(vocalbulary_jp_list, f, ensure_ascii=False)

