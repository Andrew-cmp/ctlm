import json
from make_dataset_utils import json_to_token

with open("target.json","r") as f:
  with open("append_to_train_tokenizer.json",'w') as f2:
    data = json.load(f)
    text = {}
    for target,target_info in data.items():
      target_part = " ".join(f"{key} {value} " for key, value in target_info.items())
      json.dump({'text':target_part},f2)
