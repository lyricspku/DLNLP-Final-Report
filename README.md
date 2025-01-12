# DLNLP-Final-Report

├── README.md
├── data/
│   └──data_build.json (标准日语和关西腔的平行语料数据集)
│   └──eng_jpn.txt (标准日语和英语的平行语料数据集)
│   └──test.json (标准日语和英语的翻译测试集)
│   └──train.json (标准日语和英语的翻译训练集)
│   └──valid.json (标准日语和英语的翻译验证集)
│   └──vocalbulary_en.json (英语词汇表)
│   └──vocalbulary_en_list.json (英语词汇表)
│   └──vocalbulary_jp.json (日语词汇表)
│   └──vocalbulary_jp_list.json (日语词汇表)
├── __init__.py
├── datageneration.py (标准日语和关西腔的平行语料数据集生成代码)
├── flash_attention.py
├── japanese.json
├── kansai.json
├── loss.py
├── main.py (从英语到日语的训练的transformer模型训练代码)
├── model.py
├── train.json
├── train.py
├── util.py
├── wash_data.py
