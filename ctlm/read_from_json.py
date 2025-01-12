import tvm
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.relay_integration import extracted_tasks_to_tune_contexts, extract_tasks
from tvm.meta_schedule.logging import get_loggers_from_work_dir
from typing import List, Tuple
import pickle
import argparse
import os, json, glob
from common import register_data_path,load_tasks,get_task_hashes,remove_trailing_numbers
from tqdm import tqdm  # type: ignore
import tempfile
from tvm.meta_schedule.database import JSONDatabase
from tvm import meta_schedule as ms
import numpy as np
import copy
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

    database = ms.database.JSONDatabase(path_workload=path_workload, path_tuning_record=path_tuning_record)
    all_records = database.get_all_tuning_records()

    task_name_part = None
    shape_part = []
    target_part = None

    hash_task_name = os.path.basename(work_dir)
    assert('__' in hash_task_name)
    hash, task_name = hash_task_name.split('__')
    task_name_part = task_name.split('_')

    rec = all_records[1]
    shape_list = ms.arg_info.ArgInfo.from_entry_func(rec.workload.mod, False)
    for shape in shape_list:
        shape_part.append([shape.dtype, tuple(shape.shape)])
    target_part = str(rec.target)
    print(f"target_part:{target_part}")
    trace = rec.trace
    trace.show()
    insts = trace.insts
    print(f"len(insts):{len(insts)}")
    print(f"type(insts):{type(insts)}")
    decisions = trace.decisions
    print(f"len(decisions):{len(decisions)}")
    print(f"type(decisions):{type(decisions)}")

    for decision, decisionkind in decisions.items():
        print(f"decision:{decision},decisionkind:{decisionkind}") 
    pass 
    
    return lines, path_tuning_record, path_workload, task_name_part, shape_part, target_part, hash, task_name

def read_from_json(task_path):
    data_list = []
    lines, path_tuning_record, path_workload, task_name_part, shape_part, target_part, hash, task_name = for_init_workload(task_path)
    for line, insts_part, decisions_part, decisions_label, parallel_label, latency in for_init_lines(lines, path_tuning_record):
        ppt = 'PPT'  
        data_list.append({'text': [("task_name_part",task_name_part),("shape_part",shape_part),("target_part",target_part),
                                   ("insts_part",insts_part),("decisions_part",decisions_part),
                                    ppt, 
                                    ("decisions_label",decisions_label),("parallel_label",parallel_label)]})
    with open("decision.json","w") as f:
        json.dump(data_list, f, indent=1)
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

parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
# parser.add_argument(
#     "--target",
#     type=str,
#     required=True,
# )
parser.add_argument(
    "--task_path",
    type=str,
    required=True,
)  
args = parser.parse_args()  # pylint: disable=invalid-name
read_from_json(args.task_path)

