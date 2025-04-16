import shutil
import os
import os
from tqdm import tqdm  # type: ignore
import pickle
import tempfile
from tvm import meta_schedule as ms
def get_task_info_filename(network_key, target):
    network_info_fold= "/home/houhw/tlm/tlm_dataset/meta/dataset/network_info/v100"
    assert(network_info_fold is not None)
    network_task_key = (network_key,) + ("cuda",)
    return f"{network_info_fold}/{clean_name(network_task_key)}.task.pkl"

def test_case_task():
    target = "nvidia/nvidia-v100"
    files = {

        "resnet_50": get_task_info_filename(('resnet_50', [1,3,240,240]), target),
        "mobilenet_v2": get_task_info_filename(('mobilenet_v2', [1,3,240,240]), target),
        "resnext_50": get_task_info_filename(('resnext_50', [1,3,240,240]), target),
        "bert_base": get_task_info_filename(('bert_base', [1,128]), target),
        # "gpt2": get_task_info_filename(('gpt2', [1,128]), target),
        # "llama": get_task_info_filename(('llama', [4,256]), target),
        "bert_tiny": get_task_info_filename(('bert_tiny', [1,128]), target),

        "densenet_121": get_task_info_filename(('densenet_121', [8,3,240,240]), target),
        "bert_large": get_task_info_filename(('bert_large', [4,128]), target),
        "wide_resnet_50": get_task_info_filename(('wide_resnet_50', [8,3,240,240]), target),
        "resnet3d_18": get_task_info_filename(('resnet3d_18', [4,3,128,128,16]), target),
        "dcgan": get_task_info_filename(('dcgan', [8,3,64,64]), target)
    }
    return files
def get_test_tasks():
    work_dir = tempfile.TemporaryDirectory()
    database = ms.database.JSONDatabase(work_dir=work_dir.name, module_equality='structural')
    # from common import NETWORK_INFO_FOLDER
    # files = glob.glob(f'{NETWORK_INFO_FOLDER}/*.task.pkl')
    files = test_case_task()
    files = files.values()
    extracted_tasks = []
    hash_map = {}
    for file in tqdm(files):
        with open(file, 'rb') as f:
            tasks = pickle.load(f)
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
    # work_dir.cleanup()
    # # extracted_tasks.sort(key=lambda x: x[0])
    # pickle.dump(tasks, open(f"{NETWORK_INFO_FOLDER}/all_tasks.pkl", "wb"))
    return extracted_tasks

def select_test_tasks_from_all_tasks(src_dir,dst_dir,extracted_tasks):
    hashes = get_task_hashes(extracted_tasks)

    with open("/home/houhw/ctlm/ctlm/dataset/to_test_records/failed_to_find_task_name",'w') as f:
      for task, hash in tqdm(zip(extracted_tasks, hashes)):
          ## 生成work_dir，如21225962746021776__fused_reshape_squeeze_add_reshape_transpose_broadcast_to_reshape
          dst = os.path.join(dst_dir, f'{hash}__{remove_trailing_numbers(task.task_name)}')
          selected_dir = os.path.join(src_dir,f'{hash}__{remove_trailing_numbers(task.task_name)}')
          if(os.path.isdir(selected_dir)):
            shutil.copytree(selected_dir,dst,dirs_exist_ok=True)
          else:
            f.write(f'{hash}__{remove_trailing_numbers(task.task_name)}\n')
      f.close()

def main():
  extracted_tasks = get_test_tasks()
  src_dir = "/home/houhw/tlm/tlm_dataset/meta/dataset/measure_records/v100"
  dst_dir = "/home/houhw/ctlm/ctlm/dataset/to_test_records"
  select_test_tasks_from_all_tasks(src_dir,dst_dir,extracted_tasks)

main()
