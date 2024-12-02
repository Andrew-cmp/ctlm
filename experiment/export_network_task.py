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
def _build_dataset() -> List[Tuple[str, List[int]]]:
    network_keys = []
    for name in [
        "resnet_18",
        "resnet_50",
        "mobilenet_v2",
        "mobilenet_v3",
        "wide_resnet_50",
        "resnext_50",
        "densenet_121",
        "vgg_16",
    ]:
        for batch_size in [1, 4, 8]:
            for image_size in [224, 240, 256]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    # inception-v3
    for name in ["inception_v3"]:
        for batch_size in [1, 2, 4]:
            for image_size in [299]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    # resnet3d
    for name in ["resnet3d_18"]:
        for batch_size in [1, 2, 4]:
            for image_size in [112, 128, 144]:
                network_keys.append((name, [batch_size, 3, image_size, image_size, 16]))
    # bert
    for name in ["bert_tiny", "bert_base", "bert_medium", "bert_large"]:
    # for name in ["bert_tiny", "bert_base", "bert_medium", "bert_large", "gpt2"]:
        for batch_size in [1, 2, 4]:
            for seq_length in [64, 128, 256]:
                network_keys.append((name, [batch_size, seq_length]))

    # # llama
    # for name in ["llama"]:
    #     for batch_size in [2, 4]:
    #         for seq_length in [64, 128, 256]:
    #             network_keys.append((name, [batch_size, seq_length]))

    # dcgan
    for name in ["dcgan"]:
        for batch_size in [1, 4, 8]:
            for image_size in [64]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    return network_keys


def build_network_keys():
    keys = _build_dataset()
    return keys

def dump_network():


    ## python的全局变量要这么import的吗
    network_keys = build_network_keys()
    network_key=network_keys[0]
    test_model_name,test_model_input_shape =  network_key
    
    from common import NETWORK_INFO_FOLDER
    if(os.path.exists(NETWORK_INFO_FOLDER) != 1):
        os.makedirs(NETWORK_INFO_FOLDER)
    task_info_filename=os.path.join(f"{NETWORK_INFO_FOLDER}/{test_model_name}.task.pkl")
        
    mod, params, inputs = get_network(name=test_model_name, input_shape=test_model_input_shape)
    print(mod)
    print(params)
    
    extracted_tasks=extract_tasks(
            mod,
            target,
            params,
            module_equality='structural',
            disabled_pass=None,
            instruments=None,
        )
    
    with open(task_info_filename, 'wb') as f:
            pickle.dump(extracted_tasks, f)
def get_all_tasks():
    work_dir = tempfile.TemporaryDirectory()
    database = ms.database.JSONDatabase(work_dir=work_dir.name, module_equality='structural')
    from common import NETWORK_INFO_FOLDER
    files = glob.glob(f'{NETWORK_INFO_FOLDER}/*.task.pkl')
    extracted_tasks = []
    hash_map = {}
    for file in tqdm(files):
        with open(file, 'rb') as f:
            tasks = pickle.load(f)
        filename = os.path.splitext(file)[0]
        for task_i, task in enumerate(tasks):
            hash = database.get_hash(task.dispatched[0])
            if hash in hash_map:
                if database.check_equal(hash_map[hash], task.dispatched[0]):
                    print('duplication')
                    continue
                else:
                    assert(False)
            hash_map[hash] = task.dispatched[0]
            extracted_tasks.append(task)
        with open(file, 'wb') as f:
            pickle.dump(tasks, f)
    work_dir.cleanup()
    # extracted_tasks.sort(key=lambda x: x[0])
    return extracted_tasks
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
            num_threads='physical'
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
    
    
    
def dataset_collect_models():
    dump_network()
    tasks = get_all_tasks()
    print(len(tasks))
    from common import NETWORK_INFO_FOLDER
    pickle.dump(tasks, open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "wb"))
    #dump_programs()
    

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
dump_network()
dataset_collect_models()
dump_programs()
