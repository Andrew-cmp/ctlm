from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed
import os, json, glob
import tvm.meta_schedule as ms
import copy
from make_dataset_utils import json_to_token, make_dataset
import tqdm
from multiprocessing import Pool
from tokenizer import train_tokenizer
#from meta_common import register_data_path, get_hold_out_five_files
import tvm
from functools import partial
import numpy as np
import shutil
import subprocess
import random
FOR_GEN_TOKENIZER = "for_gen_tokenizer"
FOR_GEN = "for_gen"
FOR_GEN_BEST = "for_gen_best"
FOR_GEN_BEST_ALL = "for_gen_best_all"
FOR_GEN_TRAIN_SKETCH = "for_gen_train_sketch"
FOR_GEN_EVAL_SKETCH = "for_gen_eval_sketch"
FOR_GEN_EVALTUNING_SKETCH = "for_gen_evaltuning_sketch"

# 尤其是那些主要用于存储数据的类（类似于 Java 中的 POJO 或 C++ 中的结构体）
# 使用 @dataclass，你可以让 Python 自动为类生成一些常用的特殊方法，比如 __init__、__repr__、__eq__ 等，从而减少样板代码的编写。
@dataclass
class ScriptArguments:
    # 通过 field，你可以为每个字段设置默认值、默认值生成方式、是否包含在特定方法（如 __repr__、__eq__ 等）中，甚至定义字段的元数据（metadata）。
    # field 函数来自 dataclasses 模块，它的主要作用是为数据类的字段提供额外的配置。
    for_type: str = field(metadata={"help": "", "choices": [FOR_GEN_TOKENIZER, FOR_GEN, FOR_GEN_BEST, FOR_GEN_TRAIN_SKETCH, FOR_GEN_EVAL_SKETCH, FOR_GEN_EVALTUNING_SKETCH, FOR_GEN_BEST_ALL]})
    dataset_path: str = field(metadata={"help": ""})
    tokenizer_path: str = field(metadata={"help": ""})
    target: str = field(default=None,metadata={"help": ""})
    save_path: str = field(default=None, metadata={"help": ""})
    file_cnt: int = field(default=None, metadata={"help": ""})
    keep_cnt: int = field(default=None, metadata={"help": ""})
    test_file_idx: int = field(default=None, metadata={"help": ""})
    schedule_file_path: str = field(default=None, metadata={"help": ""})
    
def get_all_dirs(dataset_path):
    all_files_and_dirs = os.listdir(dataset_path)
    all_dirs = [os.path.join(dataset_path, d) for d in all_files_and_dirs if os.path.isdir(os.path.join(dataset_path, d))]
    return all_dirs
def recursion_reset_json(json):
    assert(isinstance(json, (list, tuple)))
    for idx, it in enumerate(json):
        if isinstance(it, (list, tuple)):
            recursion_reset_json(it)
        elif isinstance(it, int):
            json[idx] = 1
        else:
            assert(False)
def for_init_workload(work_dir):
    path_tuning_record = os.path.join(work_dir, 'database_tuning_record.json')
    path_workload = os.path.join(work_dir, 'database_workload.json')
    path_tuning_record_1 = os.path.join(work_dir, 'database_tuning_record_1.json')

    with open(path_tuning_record, 'r') as f:
        lines = f.read().strip().split('\n')
    with open(path_tuning_record_1, 'w') as f:
        f.write(lines[0])
    # 相当于每个只挑了一个
    database = ms.database.JSONDatabase(path_workload=path_workload, path_tuning_record=path_tuning_record_1)
    all_records = database.get_all_tuning_records()

    task_name_part = None
    shape_part = []
    target_part = None

    hash_task_name = os.path.basename(work_dir)
    assert('__' in hash_task_name)
    hash, task_name = hash_task_name.split('__')
    task_name_part = task_name.split('_')

    for rec in all_records:
        shape_list = ms.arg_info.ArgInfo.from_entry_func(rec.workload.mod, False)
        for shape in shape_list:
            shape_part.append([shape.dtype, tuple(shape.shape)])
        target_part = str(rec.target)
    
    return lines, path_tuning_record, path_workload, task_name_part, shape_part, target_part, hash, task_name


def for_init_lines(lines, path_tuning_record):
    lines = [x for x in lines if x]
    for line in lines:
        if line == '':
            print('unexpected path_tuning_record', path_tuning_record)
        json_line = json.loads(line)

        insts_part = None
        decisions_part = []
        decisions_label = None
        parallel_label = []

        insts_part, decisions_label = json_line[1][0]
        latency = np.mean(json_line[1][1])
        for inst_i, inst in enumerate(insts_part):
            if inst[0] == 'EnterPostproc':
                break
        insts_part = insts_part[:inst_i+1]
        for inst in insts_part:
            if inst[0] == 'Annotate' and (inst[2] == ['meta_schedule.parallel']):
                parallel_label.append(copy.deepcopy(inst))
                inst[1][1] = 1
        decisions_part = copy.deepcopy(decisions_label)
        recursion_reset_json(decisions_part)
        for dec_i in range(len(decisions_part)):
            dec = decisions_part[dec_i]
            dec_label = decisions_label[dec_i]
            dec[0] = dec_label[0]
        
        yield line, insts_part, decisions_part, decisions_label, parallel_label, latency

def get_target_info(target):
    with open("target.json",'r') as f:
        target_peizhi = json.load(f)
        return target_peizhi[target]
def for_gen_tokenizer(work_dir):
    lines, path_tuning_record, path_workload, task_name_part, shape_part, target_part, hash, task_name = for_init_workload(work_dir)
    random.shuffle(lines)

    prompt_dic = {}
    virtual_target = os.path.basename(os.path.dirname(work_dir))
    target_info = get_target_info(virtual_target)
    target_part = " ".join(f"{key} {value} " for key, value in target_info.items())
    index = 0
    for line,insts_part, decisions_part, decisions_label, parallel_label, latency  in for_init_lines(lines, path_tuning_record):
        ppt = 'PPT'
        data_line = {'text': [task_name_part, shape_part, target_part, insts_part, decisions_part,
                                    ppt, 
                                    decisions_label, parallel_label],
                    'latency': latency}

        prompt_dic[index] = (latency, data_line)
        index += 1
    
    # 将所有的data_line提取出来
    prompt_dic_list = [x[1] for x in list(prompt_dic.values())]
    # 根据lable排序
    prompt_dic_list.sort(key=lambda x: x['latency'])
    for i,  data_line in enumerate(prompt_dic_list):
        data_line['text'].append(f"rank")
        data_line['text'].append(f"{i}")
        # data_line['rank'] = i
        data_line.pop("latency")
    #     print(data_line)
    # input()
    return prompt_dic_list

def process_file(args, tmp_folder, for_type, keep_cnt):
    work_dir_i, work_dir = args
    print('work_dir:', work_dir_i, ' ' * 30, end='\r')
    if for_type == FOR_GEN_TOKENIZER:
        data_list = for_gen_tokenizer(work_dir)
        data_list = json_to_token(data_list)
    else:
        assert(False)
    with open(f"{tmp_folder}/{work_dir_i}_part", "w") as f:
        for data in data_list:
            json.dump(data, f)
            f.write("\n")
            
def token_files_and_merge(for_type, dirs, save_path, keep_cnt=None):
    os.makedirs(save_path, exist_ok=True)
    
    filename = f"{save_path}/0_merge.json"
    tmp_folder = f"{save_path}/0_tmp"
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)
    with Pool() as pool:
        pool.map(partial(process_file, tmp_folder=tmp_folder, for_type=for_type, keep_cnt=keep_cnt), enumerate(dirs))
    # 这行代码使用 subprocess.run 函数在 shell 中执行一个命令。具体来说，它将 tmp_folder 目录下所有以 _part 结尾的文件内容合并到一个名为 0_merge 的文件中。
    subprocess.run(f"cat {tmp_folder}/*_part > {filename}", shell=True)
    # 将tmp_folder目录下的文件删除
    shutil.rmtree(tmp_folder)
    return filename

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    if script_args.for_type == FOR_GEN_TOKENIZER:
        all_dirs = get_all_dirs(script_args.dataset_path)
    filename = token_files_and_merge(script_args.for_type, all_dirs, script_args.tokenizer_path)
    train_tokenizer([filename], script_args.tokenizer_path, test_length=True)
main()
# python make_dataset.py \
# --for_type=for_gen_tokenizer \
# --dataset_path=/home/houhw/ctlm/ctlm/dataset/measure_records/v100_100_default_100 \
# --tokenizer_path=ctlm_data/gen_tokenizer