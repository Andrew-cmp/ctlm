from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
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

from tvm.meta_schedule.logging import get_loggers_from_work_dir

@dataclass
class ScriptArguments:
    target: str = field(metadata={"help": ""})
    ##test_file: str = field(default=None, metadata={"help": ""})
    pass


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
    print(script_args)
    register_data_path(script_args.target)
    script_args.target = tvm.target.Target(script_args.target)
    from meta_common import CURRENT_DATASET_FOLDER,HARDWARE_PLATFORM
    test_dir = f'{CURRENT_DATASET_FOLDER}/dataset/meta_schedule_tune/{HARDWARE_PLATFORM}'
    os.makedirs(test_dir,exist_ok=True)
    rand_state = ms.utils.fork_seed(None, n=1)[0]
    workloads_latency = {}
    for workload, task, hash_taskname, task_weight in yield_hold_out_five_files(script_args.target):
        work_dir = f'{test_dir}/{hash_taskname}'
        min_latency = 0

        logger = get_loggers_from_work_dir(work_dir, [task.task_name])[0]
        strategy = ms.search_strategy.EvolutionarySearch(population_size=1000,init_measured_ratio=0.0,
                                                     genetic_num_iters=1,)
        database = ms.database.JSONDatabase(work_dir=work_dir)
        ctx = ms.TuneContext(
            mod=task.dispatched[0],
            target = script_args.target,
            task_name = task.task_name,
            space_generator='post-order-apply',
            logger=logger,
            search_strategy=strategy,
            rand_state = rand_state,
            num_threads='physical'
        ).clone()
        database = ms.tune.tune_tasks(
            tasks=[ctx],
            task_weights=[task.weight],
            dataset=database,
            max_trials_global=2000,
            num_trials_per_iter = 100
        )
        if workload not in workloads_latency:
            workloads_latency[workload] = 0
        workloads_latency[workload] += float(min_latency) * task_weight
    
    val_total = 0
    for key, val in workloads_latency.items():
        print(f"{key}: {val * 1000:.4f}")
        val_total += val
    print(f"{val_total * 1000:.4f}")


if __name__ == "__main__":
    main()

