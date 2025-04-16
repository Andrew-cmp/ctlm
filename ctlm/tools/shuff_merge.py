import json
import os
import random
path = "/home/houhw/ctlm/tlm_dataset/meta/meta_data/a100_tokenizer/0_merge.json"
path = "/home/houhw/ctlm/tlm_dataset/meta/meta_data/v100_gen/0_merge.json"

with open(path, 'r') as f:
    lines = f.read().strip().split('\n')
lines = [x for x in lines]
random.shuffle(lines)
with open(path, 'r') as f:
    for data in lines:
        json.dump(data, f)
        f.write("\n")
print(len(lines))
max_len = -1
max_len_str : str
for line in lines:
    json_line = json.loads(line)
    text = json_line['text']
    text = text.strip().split(' ')
    if(len(text) > max_len):
        max_len = len(text)
        max_len_str = text
print(max_len)
print(max_len_str)
    
    