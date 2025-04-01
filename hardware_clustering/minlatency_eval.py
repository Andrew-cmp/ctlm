from dataclasses import dataclass, field
from typing import Optional
import argparse
import json
from tvm import auto_scheduler
import tvm
import numpy as np
import tqdm
import pickle
import os
from meta_common import register_data_path, hold_out_task_files, get_jsondatabase_top1
import tvm.meta_schedule as ms
from meta_common import yield_hold_out_five_files
from glob import glob
import shutil
import logging
from tvm.target import Target
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--for_test", type=bool,default=False ,help="measure test_dir candidate on target."
    )    
    parser.add_argument(
        "--for_tune", type=bool,default=False ,help="tune test_dir mod on target"
    )
    parser.add_argument(
        "--candidate_cache_dir", type=str, help="Please provide the full path to the candidates."
    )
    parser.add_argument(
        "--result_cache_dir", type=str, help="Please provide the full path to the result database."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="nvidia/nvidia-a6000",
        help="Please specify the target hardware for tuning context.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size of candidates sent to builder and runner each time.",
    )
    return parser.parse_args()
script_args= _parse_args()

# def read_gen_best_json(json_path):
#     with open(json_path, "r") as f:
#         lines = f.read().strip().split("\n")
#     min_latency_dict = {}
#     for line in lines:
#         json_line = json.loads(line)
#         latency = json_line["latency"]
#         json_line = json.loads(json_line["note"]["line"])
#         workload_key = json_line["i"][0][0]
#         if workload_key not in min_latency_dict:
#             min_latency_dict[workload_key] = 1e10
#         min_latency_dict[workload_key] = min(min_latency_dict[workload_key], latency)
#     return min_latency_dict


# def read_fine_tuning_json(json_path):
#     with open(json_path, "r") as f:
#         lines = f.read().strip().split("\n")
#     min_latency_dict = {}
#     for line in lines:
#         json_line = json.loads(line)
#         latencies = json_line["r"][0]
#         latency = sum(latencies) / len(latencies)
#         workload_key = json_line["i"][0][0]
#         if workload_key not in min_latency_dict:
#             min_latency_dict[workload_key] = 1e10
#         min_latency_dict[workload_key] = min(min_latency_dict[workload_key], latency)
#     return min_latency_dict


# def best_lines_convert():
#     from common import HARDWARE_PLATFORM
#     with open(f'gen_data/{HARDWARE_PLATFORM}_gen_best/0_merge.json', 'r') as f:
#         lines = f.read().strip().split('\n')

#     target_path = f'gen_data/measure_data_{HARDWARE_PLATFORM}/best.json'
#     with open(target_path, 'w') as f:
#         for line in lines:
#             f.write(json.loads(line)['line'])
#             f.write('\n')
#     return target_path
def tuning_mod(database,result_dir):
    
    workload = database.get_all_tuning_records()[0].workload
    mod = workload.mod
    model_name, workload_name = database.path_workload.split("/")[-2:]
    record_name = database.path_tuning_record.split("/")[-1]

    database = ms.tir_integration.tune_tir(
        mod=mod,
        target=script_args.target,
        work_dir=result_dir,
        max_trials_global=2000,
        num_trials_per_iter=500,
    )
    
def measure_candidates(database, builder, runner, task_record,result_dir):
    """Send the candidates to builder and runner for distributed measurement,
    and save the results in a new json database.

    Parameters
    ----------
    database : JSONDatabase
        The database for candidates to be measured.
    builder : Builder
        The builder for building the candidates.
    runner : Runner
        The runner for measuring the candidates.

    Returns
    -------
    None
    """
    candidates, runner_results, build_fail_indices, run_fail_indices = [], [], [], []
    tuning_records = database.get_all_tuning_records()
    if len(tuning_records) == 0:
        return
    for record in tuning_records:
        candidates.append(record.as_measure_candidate())
    model_name, workload_name = database.path_workload.split("/")[-2:]
    record_name = database.path_tuning_record.split("/")[-1]
    #candidates = add_candidates_func_attr(candidates,model_name)
    with ms.Profiler() as profiler:
        for idx in range(0, len(candidates), script_args.batch_size):
            batch_candidates = candidates[idx : idx + script_args.batch_size]
            task_record._set_measure_candidates(batch_candidates)  # pylint: disable=protected-access
            with ms.Profiler.timeit("build"):
                task_record._send_to_builder(builder)  # pylint: disable=protected-access
            with ms.Profiler.timeit("run"):
                task_record._send_to_runner(runner)  # pylint: disable=protected-access
                batch_runner_results = task_record._join()  # pylint: disable=protected-access
            runner_results.extend(batch_runner_results)
            for i, result in enumerate(task_record.builder_results):
                if result.error_msg is None:
                    ms.utils.remove_build_dir(result.artifact_path)
                else:
                    build_fail_indices.append(i + idx)
            task_record._clear_measure_state(batch_runner_results)  # pylint: disable=protected-access

    new_database = ms.database.JSONDatabase(
        path_workload=os.path.join(result_dir, model_name, workload_name),
        path_tuning_record=os.path.join(result_dir, model_name, record_name),
    )
    workload = tuning_records[0].workload
    new_database.commit_workload(workload.mod)
    for i, (record, result) in enumerate(zip(tuning_records, runner_results)):
        new_database.commit_tuning_record(
            ms.database.TuningRecord(
                trace=record.trace,
                workload=workload,
                run_secs=[v.value for v in result.run_secs],
                target=Target(script_args.target),
            )
        )

    fail_indices_name = workload_name.replace("_workload.json", "_failed_indices.txt")
    build_fail_indices_name = workload_name.replace("_workload.json", "_build_failed_indices.txt")
    with open(
        os.path.join(result_dir, model_name, fail_indices_name), "w", encoding="utf8"
    ) as file:
        file.write(" ".join([str(n) for n in run_fail_indices]))
    with open(
        os.path.join(result_dir, model_name, build_fail_indices_name), "w", encoding="utf8"
    ) as file:
        file.write(" ".join([str(n) for n in build_fail_indices]))
    logging.info(
        f"Builder time: {profiler.get()['build']}, Runner time: {profiler.get()['run']}\n\
            Build model is {model_name}\n\
            Failed number of builds: {len(build_fail_indices)},\
            Failed number of runs: {len(run_fail_indices)}"
    )
def get_target_name(target_str):
    model_list = ['i7', 'v100', 'a100', '2080', 'None','a6000','Xeon-Platinum-8488C']
    for model in model_list:
        if model in target_str:
            break
    assert(model != 'None')
    return model
def main():
    print(script_args)
    print(script_args.target)

    print(type(script_args.target))
    # # Load task registry
    # print("Load all tasks...")
    # tasks = load_and_register_tasks()
    # register_data_path(script_args.target)
    # script_args.target = tvm.target.Target(script_args.target)

    # if script_args.for_train:
    #     assert(False)
    #     # test_file = best_lines_convert()
    #     # from gen_utils import get_finetuning_files
    #     # print('get_testtuning_files', get_finetuning_files()[-1])
    # elif script_args.for_test:
    #     from meta_utils import get_test_dirs
    #     test_dir = get_test_dirs()[-1]
    # elif script_args.for_testtuning:
    #     from meta_common import HARDWARE_PLATFORM
    #     test_dir = f'dataset/measure_records/{HARDWARE_PLATFORM}'
    #     from meta_utils import get_testtuning_dirs
    #     print('get_testtuning_dirs', get_testtuning_dirs()[-1])
    # elif script_args.test_file:
    #     assert(False)
    #     # test_file = script_args.test_file
    # else:
    #     assert(False)
    candidate_cache_dir = script_args.candidate_cache_dir
    print('-' * 50)
    print("candidate_cache_dir dir:", script_args.candidate_cache_dir)
    
    workloads_latency = {}
    work_dirs = glob(os.path.join(candidate_cache_dir,'*'))
    if(script_args.for_test == True):
        builder = ms.builder.LocalBuilder(timeout_sec=3000)
        runner = ms.runner.LocalRunner(timeout_sec=100)
        task_record = ms.task_scheduler.task_scheduler.TaskRecord(
        ms.TuneContext(target=Target(script_args.target)))
        
        for work_dir in work_dirs:
            model_name = work_dir.split("/")[-1]
            result_dir = script_args.result_cache_dir
            if(os.path.exists(os.path.join(result_dir,model_name))):
                pass
            else:
                os.makedirs(os.path.join(result_dir,model_name))
                database = ms.database.JSONDatabase(work_dir=work_dir)
                try:
                    measure_candidates(database,builder,runner,task_record,result_dir)
                except Exception as e:
                    print(f"{model_name} has except:{e},{model_name} dir has been removed")
                    shutil.rmtree(os.path.join(result_dir,model_name))
                
    elif(script_args.for_tune == True):
        builder = ms.builder.LocalBuilder(timeout_sec=3000)
        runner = ms.runner.LocalRunner(timeout_sec=100)
        task_record = ms.task_scheduler.task_scheduler.TaskRecord(
        ms.TuneContext(target=Target(script_args.target)))
        for work_dir in work_dirs:
            model_name = work_dir.split("/")[-1]
            result_dir = script_args.result_cache_dir
            if(os.path.exists(os.path.join(result_dir,model_name))):
                pass
            else :
                os.makedirs(os.path.join(result_dir,model_name))
                database = ms.database.JSONDatabase(work_dir=work_dir)
                saved_dir = os.path.join(result_dir,model_name)
                tuning_mod(database,saved_dir)
    else:
        for work_dir in work_dirs:
            min_latency, _ = get_jsondatabase_top1(work_dir)
            task_name = os.path.basename(work_dir)
            if task_name not in workloads_latency:
                workloads_latency[task_name] = 0
            workloads_latency[task_name] = float(min_latency) 
            print(f"task_name:{task_name} min_latency:{min_latency}")


if __name__ == "__main__":
    main()
# python minlatency_eval.py \
# --for_test=true \
# --candidate_cache_dir=dataset/test_tmp_to_measure \
# --result_cache_dir=dataset/test_tmp_measured \
# --target=nvidia/nvidia-v100 

# PYTHONUNBUFFERED=1 python -u minlatency_eval.py \
# --for_test=true \
# --candidate_cache_dir=dataset/top_random_sample \
# --result_cache_dir=dataset/subgragh_4_hc_measured \
# --target=nvidia/nvidia-v100 >run.log 2>&1



