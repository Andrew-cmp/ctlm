
import argparse
import os
import tvm.meta_schedule as ms
import random
from glob import glob
import shutil
import logging
from tvm.target import Target
##从candidate中选择子图中的部分调度用以硬件建模
##有topk策略和随机策略
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate_cache_dir", type=str, help="Please provide the full path to the candidates."
    )
    parser.add_argument(
        "--result_cache_dir", type=str, help="Please provide the full path to the result database."
    )
    parser.add_argument(
        "--k", type=int, required=True ,help="Pick out k "
    )    
    parser.add_argument(
        "--target", type=str, help="Pick out the top k"
    )    
    parser.add_argument(
        "--choice", required=True,type=str, choices = ["top_k","random","top_random"],help="Pick out the top k"
    )
    return parser.parse_args()
script_args= _parse_args()
def top_k(work_dirs,result_path):
    for work_dir in (work_dirs):
        model_name = work_dir.split("/")[-1]
        old_database = ms.database.JSONDatabase(work_dir=work_dir)
        tuning_records = old_database.get_all_tuning_records()
        if(len(tuning_records)<script_args.k):
            print(
                f"workload :{model_name} has record num {len(tuning_records)},less than {script_args.k}")
            continue
        workload = tuning_records[0].workload
        top_k_tuning_recoreds = old_database.get_top_k(workload,script_args.k)

        model_name, workload_name = old_database.path_workload.split("/")[-2:]
        record_name = old_database.path_tuning_record.split("/")[-1]
        
        new_dir = os.path.join(result_path, model_name)
        os.makedirs(new_dir,exist_ok=False)
        
        new_database = ms.database.JSONDatabase(
            path_workload=os.path.join(result_path, model_name, workload_name),
            path_tuning_record=os.path.join(result_path, model_name, record_name),)
        
        new_database.commit_workload(workload.mod)
        for record in top_k_tuning_recoreds:
            new_database.commit_tuning_record(
                ms.database.TuningRecord(
                    trace=record.trace,
                    workload=workload,
                    target=Target(script_args.target)
                )
            )
        print(
            f"workload :{model_name} has record num {len(tuning_records)}, pick top {len(top_k_tuning_recoreds)} from them"
        )
def random_choice(work_dirs,result_path):
    for work_dir in (work_dirs):
        model_name = work_dir.split("/")[-1]
        old_database = ms.database.JSONDatabase(work_dir=work_dir)
        tuning_records = old_database.get_all_tuning_records()
        if(len(tuning_records)<script_args.k):
            print(
                f"workload :{model_name} has record num {len(tuning_records)},less than {script_args.k}")
            continue
        tuning_records_list = [i for i in tuning_records]
        chosed_tuning_record = random.sample(tuning_records_list, script_args.k)
        
        model_name, workload_name = old_database.path_workload.split("/")[-2:]
        record_name = old_database.path_tuning_record.split("/")[-1]
        new_dir = os.path.join(result_path, model_name)
        os.makedirs(new_dir,exist_ok=False)

        new_database = ms.database.JSONDatabase(
            path_workload=os.path.join(result_path, model_name, workload_name),
            path_tuning_record=os.path.join(result_path, model_name, record_name),)

        workload = tuning_records[0].workload
        new_database.commit_workload(workload.mod)
        for record in chosed_tuning_record:
            new_database.commit_tuning_record(
                ms.database.TuningRecord(
                    trace=record.trace,
                    workload=workload,
                    target=Target(script_args.target)
                )
            )
        print(
            f"workload :{model_name} has record num {len(tuning_records)}, pick top {len(chosed_tuning_record)} from them"
        )
def top_random(work_dirs,result_path):
    print(f"ours choice is {script_args.k}")
    for work_dir in (work_dirs):
        model_name = work_dir.split("/")[-1]
        old_database = ms.database.JSONDatabase(work_dir=work_dir)
        tuning_records = old_database.get_all_tuning_records()
        workload = tuning_records[0].workload
        top_k_tuning_recoreds = old_database.get_top_k(workload,len(tuning_records)//2)
        if(len(top_k_tuning_recoreds)<script_args.k):
            print(
                f"workload :{model_name} has record num {len(tuning_records)},we choice top half, but less than {script_args.k}")
            continue
        tuning_records_list = [i for i in top_k_tuning_recoreds]
        chosed_tuning_record = random.sample(tuning_records_list, script_args.k)
        
        model_name, workload_name = old_database.path_workload.split("/")[-2:]
        record_name = old_database.path_tuning_record.split("/")[-1]
        new_dir = os.path.join(result_path, model_name)
        os.makedirs(new_dir,exist_ok=False)

        new_database = ms.database.JSONDatabase(
            path_workload=os.path.join(result_path, model_name, workload_name),
            path_tuning_record=os.path.join(result_path, model_name, record_name),)

        new_database.commit_workload(workload.mod)
        for record in chosed_tuning_record:
            new_database.commit_tuning_record(
                ms.database.TuningRecord(
                    trace=record.trace,
                    workload=workload,
                    target=Target(script_args.target)
                )
            )
        print(
            f"workload :{model_name} has record num {len(tuning_records)}, random pick {len(chosed_tuning_record)} from top half"
        )
def main():
    work_path = script_args.candidate_cache_dir
    result_path = script_args.result_cache_dir

    work_dirs = glob(os.path.join(work_path,'*'))
    if(script_args.choice == "top_k"):
        top_k(work_dirs,result_path)
    elif(script_args.choice == "random"):
        random_choice(work_dirs,result_path)
    elif(script_args.choice == "top_random"):
        top_random(work_dirs,result_path)
    
if __name__ == "__main__":
    main()
# python -u select_trace.py --candidate_cache_dir=dataset/subgragh_4_hc_to_measure \
# --result_cache_dir=dataset/tmp_random \
# --target=nvidia/nvidia-v100 \
# --choice=random \
# --k=200 > select_trace.log 2>&1

# python -u select_trace.py --candidate_cache_dir=dataset/origin_dataset \
# --result_cache_dir=dataset/top_random_sample \
# --target=nvidia/nvidia-v100 \
# --choice=top_random \
# --k=50 > select_trace.log 2>&1