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


def _args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lhs", 
        type=str,
        required=True,
        help="Please specify the lhs json file or directory containing json files.",
    )
    parser.add_argument(
        "--rhs",
        type=str,
        required=True,
        help="Please specify the rhs json file or directory containing json files.",
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
def compare(file, lhs_path, rhs_path):
    lhs_file_path = os.path.join(lhs_path, file)
    rhs_file_path = os.path.join(rhs_path, file)
    
    lhs_json_lines = []
    with open(lhs_file_path, 'r') as f:
        lhs_lines = f.read().strip().split('\n')
        lhs_lines = [x for x in lhs_lines if x]
        for lhs_line in lhs_lines:
            lhs_json_lines.append(json.loads(lhs_line))
            
    rhs_json_lines = []
    with open(rhs_file_path, 'r') as f:
        rhs_lines = f.read().strip().split('\n')
        rhs_lines = [x for x in rhs_lines if x]
        for rhs_line in rhs_lines:
            rhs_json_lines.append(json.loads(rhs_line))
            
    print(f"len of lhs {file} is {len(lhs_lines)}")
    print(f"len of rhs {file} is {len(rhs_lines)}")
    # 只需要前十个
    num = 10    
    t = 0
    l_r = []
    for index_l,lhs_line in enumerate(lhs_json_lines):
        if t > num:
            break
        else:
            t += 1
        l_line_i = lhs_line['line_i']
        l_rank = index_l
        is_find = False
        for index_r, rhs_line in enumerate(rhs_json_lines) :
            r_line_i = rhs_line['line_i']
            r_rank = index_r
            if(r_line_i == l_line_i):
                l_r.append({'l_rank':l_rank,'r_rank':r_rank,'line_i':r_line_i})
                is_find = True
                break
        if not is_find:
            print(f"can not find {l_line_i} in {file}")
               
    for t in l_r:
        print(f"lhs line_i:{t['line_i']} rank rise:{(t['l_rank']-t['r_rank'])}")
     
    
args = _args_parser()
lhs = args.lhs
rhs = args.rhs
rhs_files_name = []
lhs_flies_name = []
if os.path.isdir(rhs):
    rhs_files_name = os.listdir(rhs)
    rhs_path = rhs
elif os.path.isfile(rhs):
    if rhs.endswith(".json"):
        rhs_files_name = [os.path.basename(rhs)] 
        rhs_path = os.path.dirname(rhs)
    else: 
        print("rhs file should be a json type")
        exit(1)
else:
    print("rhs should be a json file or a directory containing json files")
    exit(1)
    
if os.path.isdir(lhs):
    lhs_flies_name =os.listdir(lhs)
    lhs_path = lhs
elif os.path.isfile(lhs):
    if lhs.endswith(".json"):
        lhs_flies_name = [os.path.basename(lhs)] 
        lhs_path = os.path.dirname(lhs)
    else: 
        print("lhs file should be a json type")
        exit(1)
else:
    print("lhs should be a json file or a directory containing json files")
    exit(1)

compare_list = []

for lhs_file_name in lhs_flies_name:
    if lhs_file_name not in rhs_files_name:
        print(f"{lhs_file_name} not in {rhs}")
        continue
    else:
        compare_list.append(lhs_file_name)
for file in compare_list:
    print('*'*15)
    print(f"{file} will be compared")
    print('*'*15)
    compare(file, lhs_path,rhs_path)
#python compare_json.py --lhs=/home/houhw/tlm/experiment/dataset/ranked_records/a6000_53/2303760369153750735__fused_nn_conv2d_add.json --rhs=/home/houhw/tlm/experiment/dataset/ranked_records/a6000_100/2303760369153750735__fused_nn_conv2d_add.json
#python compare_json.py --lhs=/home/houhw/tlm/experiment/dataset/ranked_records/a6000_53/ --rhs=/home/houhw/tlm/experiment/dataset/ranked_records/a6000_100/
