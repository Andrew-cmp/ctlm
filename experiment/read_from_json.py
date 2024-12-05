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

def read_from_jason(task):
    from common import TO_MEASURE_PROGRAM_FOLDER
    path_workload = TO_MEASURE_PROGRAM_FOLDER+'/'+task+'/'+"database_workload.json"
    path_tuning_record = TO_MEASURE_PROGRAM_FOLDER+'/'+task+'/'+"database_tuning_record.json"
    dateset = JSONDatabase(path_tuning_record=path_tuning_record,path_workload=path_workload)
    all_records = dateset.get_all_tuning_records()
    task_name_part = None
    shape_part = []
    target_part = None

    hash_task_name = task
    assert('__' in hash_task_name)
    hash, task_name = hash_task_name.split('__')
    task_name_part = task_name.split('_')

    for rec in all_records:
        shape_list = ms.arg_info.ArgInfo.from_entry_func(rec.workload.mod, False)
        for shape in shape_list:
            shape_part.append([shape.dtype, tuple(shape.shape)])
        target_part = str(rec.target)
        # trace = rec.trace
        # # 这种可读性不如trace.show的可读性。
        # print(trace.as_python())
        # trace.show()
        # insts = trace.insts
        # print(f"len(insts):{len(insts)}")
        # print(f"type(insts):{type(insts)}")
        # decisions = trace.decisions
        # print(f"len(decisions):{len(decisions)}")
        # print(f"type(decisions):{type(decisions)}")
        
        # for inst in insts:
        #     print(f"inst:{inst}")
        # for decision, decisionkind in decisions.items():
        #     print(f"decision:{decision},decisionkind:{decisionkind}") 
        # pass 
    return  path_tuning_record, path_workload, task_name_part, shape_part, target_part, hash, task_name


parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
parser.add_argument(
    "--target",
    type=str,
    required=True,
)    
args = parser.parse_args()  # pylint: disable=invalid-name
target = tvm.target.Target(args.target)
register_data_path(args.target)
from common import TO_MEASURE_PROGRAM_FOLDER
tasks = os.listdir(TO_MEASURE_PROGRAM_FOLDER)
for task in tasks:
    read_from_jason(task)

