import tvm
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.relay_integration import extracted_tasks_to_tune_contexts, extract_tasks
from tvm.meta_schedule.logging import get_loggers_from_work_dir
from typing import List, Tuple
import pickle
import argparse
import os, json, glob
from common import register_data_path,load_tasks,get_task_hashes,remove_trailing_numbers
import tempfile

from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
# pylint: disable=too-many-branches

def _dump_programs(part_extracted_tasks, work_i, dir_path):
    hashes = get_task_hashes(part_extracted_tasks)
    for task, hash in tqdm(zip(part_extracted_tasks, hashes)):
        ## 生成work_dir，如21225962746021776__fused_reshape_squeeze_add_reshape_transpose_broadcast_to_reshape
        work_dir = os.path.join(dir_path, f'{hash}__{remove_trailing_numbers(task.task_name)}')
        if os.path.exists(work_dir):
            with open(os.path.join(work_dir, 'database_tuning_record.json'), 'r') as f:
                lines = [line for line in f.read().strip().split('\n') if line]
                if len(lines) > 0:
                    continue

        logger = get_loggers_from_work_dir(work_dir, [task.task_name])[0]
        rand_state = ms.utils.fork_seed(None, n=1)[0]

        ctx = ms.TuneContext(
            mod=task.dispatched[0],
            target=task.target,
            space_generator='post-order-apply',
            search_strategy=ms.search_strategy.EvolutionarySearch(population_size=1000),
            task_name=task.task_name,
            logger=logger,
            rand_state=rand_state,
            num_threads='logical'
        ).clone()

        ms.tune.dump_program(
            tasks=[ctx],
            task_weights=[task.weight],
            work_dir=work_dir,
            max_trials_global=2048
        )
    
def dump_programs():
    
    all_extracted_tasks = load_tasks()
    print('len extracted tasks:', len(all_extracted_tasks))
    from common import HARDWARE_PLATFORM
    dir_path = f'dataset/to_measure_programs/{HARDWARE_PLATFORM}'
    os.makedirs(dir_path, exist_ok=True)
    _dump_programs(all_extracted_tasks,dir_path,dir_path)
    

parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
parser.add_argument(
    "--target",
    type=str,
    required=True,
)    
args = parser.parse_args()  # pylint: disable=invalid-name
target = tvm.target.Target(args.target)
register_data_path(args.target)
from common import NETWORK_INFO_FOLDER
path_network_info_folder = os.path.join(f'{NETWORK_INFO_FOLDER}',args.target)
#dataset_collect_models()
dump_programs()
