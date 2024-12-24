import json
input_file = 'train.json'
output_file = 'japanese.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

japanese_sentences = [item['text_jp'] for item in data]

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(japanese_sentences, f, ensure_ascii=False, indent=4)

print(f"save file {output_file}")
