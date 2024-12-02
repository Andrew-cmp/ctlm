from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed
import os, json, glob
import tvm.meta_schedule as ms
import copy
import tqdm
from multiprocessing import Pool
import tvm
from functools import partial
import numpy as np
import shutil
import subprocess
import random
import argparse
from common import register_data_path,load_tasks,get_task_hashes,remove_trailing_numbers
def _args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", 
        type=str,
        default="nvidia/nvidia-a6000",
        help="Please specify the target hardware for tuning context.",
    )
    # parser.add_argument(
    #     "--work_dir",
    #     type=str,
    #     required=True,
    #     help="Please provide the full path to the mearsured records."
    # )
    # parser.add_argument(
    #     "--save_path",
    #     type=str,
    #     required=True,
    #     help="Please provide the full path to the ranked records."
    # )
    return parser.parse_args()
def recursion_reset_json(json):
    assert(isinstance(json, (list, tuple)))
    for idx, it in enumerate(json):
        if isinstance(it, (list, tuple)):
            recursion_reset_json(it)
        elif isinstance(it, int):
            json[idx] = 1
        else:
            assert(False)
# database_tuning_record_1.json在这里
def for_init_workload(work_dir):
    path_tuning_record = os.path.join(work_dir, 'database_tuning_record.json')
    path_workload = os.path.join(work_dir, 'database_workload.json')

    with open(path_tuning_record, 'r') as f:
        lines = f.read().strip().split('\n')
    # 相当于每个只挑了一个
    database = ms.database.JSONDatabase(path_workload=path_workload, path_tuning_record=path_tuning_record)
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
def get_ranked_list_from_json(work_dir):
    data_lists = []
    lines, path_tuning_record, path_workload, task_name_part, shape_part, target_part, hash, task_name = for_init_workload(work_dir)
    for line_i, ret in enumerate(for_init_lines(lines, path_tuning_record)):
        line, insts_part, decisions_part, decisions_label, parallel_label, latency = ret
        data_lists.append({'text': [task_name_part, shape_part, target_part, insts_part, decisions_part,
                                    decisions_label, parallel_label],
                           'latency': latency,
                           'line_i': line_i})
    data_lists = sorted(data_lists, key=lambda x: x['latency'])
    return data_lists



def ranked_records_to_json(work_dir, save_path, keep_cnt=None):
    os.makedirs(save_path, exist_ok=True)
    work_dirs = get_all_dirs(work_dir)
    for i,work_dir in enumerate(work_dirs):
        process_file([i,work_dir], save_path)
    # with Pool() as pool:
    #     pool.map(partial(process_file, save_path=save_path), enumerate(work_dirs))
    # print()
    # 这行代码使用 subprocess.run 函数在 shell 中执行一个命令。具体来说，它将 tmp_folder 目录下所有以 _part 结尾的文件内容合并到一个名为 0_merge 的文件中。

def process_file(args, save_path):
    work_dir_i, work_dir = args
    print('work_dir:', work_dir_i, ' ' * 30, end='\r')
    data_list = get_ranked_list_from_json(work_dir)
    #data_list = json_to_token(data_list)
    save_path = os.path.join(save_path, os.path.basename(work_dir))
    with open(f"{save_path}", "w") as f:
        for data in data_list:
            json.dump(data, f)
            f.write("\n")
# def range_jason(work_dir,save_path):
#     all_dirs = []
#     all_dirs.extend(get_all_dirs(work_dir))
#     hash_task_name = os.path.basename(work_dir)
#     save_path = os.path.join(save_path, hash_task_name)
#     os.makedirs(save_path, exist_ok=True)
#     path_tuning_record = os.path.join(work_dir, 'database_tuning_record.json')
#     path_workload = os.path.join(work_dir, 'database_workload.json')
#     with open(path_tuning_record, 'r') as f:
#         lines = f.read().strip().split('\n')
#     database = ms.database.JSONDatabase(path_workload=path_workload, path_tuning_record=path_tuning_record)
#     all_records = database.get_all_tuning_records()
#     hash, task_name = hash_task_name.split('__')
#     task_name_part = task_name.split('_')
    
            
def get_all_dirs(dataset_path):
    all_files_and_dirs = os.listdir(dataset_path)
    all_dirs = [os.path.join(dataset_path, d) for d in all_files_and_dirs if os.path.isdir(os.path.join(dataset_path, d))]
    return all_dirs

def main():
    args = _args_parser()
    target = args.target
    register_data_path(target)
    from common import MEASURE_RECORD_FOLDER
    from common import RANKED_RECORD_FOLDER 
    ranked_records_to_json(MEASURE_RECORD_FOLDER,RANKED_RECORD_FOLDER,keep_cnt=None)
    
    
if __name__ == "__main__":
    main()
# python rank.py --target=nvidia/nvidia-a6000