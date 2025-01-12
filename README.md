# DLNLP-Final-Report

├── README.md <br>
├── data/ <br>
│   └──data_build.json (标准日语和关西腔的平行语料数据集) <br>
│   └──eng_jpn.txt (标准日语和英语的平行语料数据集) <br>
│   └──test.json (标准日语和英语的翻译测试集) <br>
│   └──train.json (标准日语和英语的翻译训练集) <br>
│   └──valid.json (标准日语和英语的翻译验证集) <br>
│   └──vocalbulary_en.json (英语词汇表) <br>
│   └──vocalbulary_en_list.json (英语词汇表) <br>
│   └──vocalbulary_jp.json (日语词汇表) <br>
│   └──vocalbulary_jp_list.json (日语词汇表) <br>
├── __init__.py <br>
├── datageneration.py (标准日语和关西腔的平行语料数据集生成代码) <br>
├── flash_attention.py <br>
├── japanese.json <br>
├── kansai.json <br>
├── loss.py <br>
├── main.py (从英语到日语的训练的transformer模型训练代码) <br>
├── model.py <br>
├── train.json <br>
├── train.py <br>
├── util.py <br>
├── wash_data.py <br>
