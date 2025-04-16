
import argparse
import os, json

# 打开tokenizer使用merge.json文件，查看里面的prompt的最大长度是多少
# 确实仍然是589
def read_from_json(task_path):
    max_len = -1
    max_s:str
    with open(task_path, 'r') as f:
        lines = f.read().strip().split("\n")
    for line in lines:
        try:
            text = json.loads(line)["text"]
            s = text.split(" ")
            l = len(s)
            max_len = max(max_len,l)
            if(l == max_len):
                max_s  = text
        except:
            text = line
    print(max_len)
    print(max_s)
parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
parser.add_argument(
    "--task_path",
    type=str,
    required=True,
)  
args = parser.parse_args()  # pylint: disable=invalid-name
read_from_json(args.task_path)

#python find_maxlen_from_mergejson.py --task_path=/home/houhw/ctlm/ctlm/ctlm_data/ctlm_tokenizer/0_merge.json