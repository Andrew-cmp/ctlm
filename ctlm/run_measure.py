import os, glob, time
from multiprocessing import Process, Lock
from urllib.parse import quote
import math
import shutil
import re
import argparse
n_part = 4
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reg_times", type=float, help="this vaule is related to mps tool."
    )
    parser.add_argument(
        "--cuda_id1", type=str, help="this vaule is related to mps tool.",default="-1"
    )
    parser.add_argument(
        "--cuda_id2", type=str, help="this vaule is related to mps tool.",default="-1"
    )
    return parser.parse_args()
args = _parse_args()
def exec_cmd_if_error_send_mail(command):
    print("#" * 50)
    print("command:", command)
    returncode = os.system(command)
    print("returncode:", returncode)
    return returncode


#这段代码的整体目的是：
#检查待处理的目录列表，如果有待处理的目录则开始处理。
#将待处理目录下的文件分割成若干部分，每部分创建一个新的目录。
#将每部分的文件复制到相应的新目录。
#备份原始的待处理目录，将其移动到备份位置。
# 将to_measure下面的目录分割成n_part个部分，分别为a6000_part_0,a6000_part_1,a6000_part_2等。
def divide_worker(lock):
    while True:
        to_measure_list = glob.glob("./measure_data/to_measure/*")
        ## 对 to_measure_list 进行过滤，移除那些名称中包含 _part_ 或 zip 的文件或目录，确保只处理特定类型的文件。
        ## to_measure_list包含fintuning_0123
        to_measure_list = [x for x in to_measure_list if '_part_' not in x and 'zip' not in x]
        ## 对过滤后的 to_measure_list 进行排序，以确保以一致的顺序处理目录。
        to_measure_list.sort()
        if len(to_measure_list) > 0:
            with lock:
                time.sleep(10)
                to_measure_dir = to_measure_list[0]
                tasks = glob.glob(f'{to_measure_dir}/*')
                part_len = math.ceil(len(tasks) / n_part)
                for i in range(n_part):
                    dir_i = f"{to_measure_dir}_part_{i}"
                    os.makedirs(dir_i)
                    tasks_part = tasks[i*part_len : (i+1)*part_len]
                    for task in tasks_part:
                        source_dir = task
                        target_dir = source_dir.replace(to_measure_dir, dir_i)
                        shutil.copytree(source_dir, target_dir)
                to_measure_bak_dir = os.path.join("measure_data/to_measure_bak", os.path.basename(to_measure_dir))
                command = f"rm -rf {to_measure_bak_dir}; mv {to_measure_dir} {to_measure_bak_dir}"
                exec_cmd_if_error_send_mail(command)
        time.sleep(10)


def merge_worker(lock):
    while True:
        measure_part_list = glob.glob("measure_data/measured_part/*_part_*")
        for part_0 in measure_part_list:
            if '_part_0' in part_0:
                finish = True
                merge_dir_list = [part_0]
                for i in range(1, n_part, 1):
                    part_i = f'{part_0[:-1]}{i}'
                    if part_i not in measure_part_list:
                        finish = False
                        break
                    merge_dir_list.append(part_i)
                if finish:
                    with lock:
                        measured_part_dir = os.path.join("measure_data/measured_part", os.path.basename(part_0[:-len('_part_0')]))
                        for dir in merge_dir_list:
                            tasks = glob.glob(f'{dir}/*')
                            for task in tasks:
                                src = task
                                dest = task.replace(dir, measured_part_dir)
                                shutil.move(src, dest)
                            shutil.rmtree(dir)
                        measured_dir = os.path.join("measure_data/measured", os.path.basename(part_0[:-len('_part_0')]))
                        src = measured_part_dir
                        dest = measured_dir
                        shutil.move(src, dest)
        time.sleep(10)


def worker(gpu_id, lock):
    time.sleep(9 - 2)

    while True:
        with lock:
            to_measure_list = glob.glob("measure_data/to_measure/*_part_*")

            to_measure_list.sort()
            to_measure_file = None
            if len(to_measure_list) > 0:
                to_measure_file = to_measure_list[0]

                to_measure_dir = f"measure_data/to_measure_{gpu_id}"
                os.makedirs(to_measure_dir, exist_ok=True)
                to_measure_dir_file = os.path.join(to_measure_dir, os.path.basename(to_measure_file))
                exec_cmd_if_error_send_mail(f"mv {to_measure_file} {to_measure_dir_file}")

        if to_measure_file:
            base_name = os.path.basename(to_measure_file)
            match = re.search(r'part_(\d+)', base_name)
            part_id = match.group(1)
            measured_tmp_file = os.path.join("measure_data/measured_tmp", os.path.basename(to_measure_file))
            print(f"part_id is :{part_id}")
            moved_dir = "measure_data/moved"
            while True:
                command = f"CUDA_VISIBLE_DEVICES={gpu_id} python measure_programs.py --result_error_threshold=5 --reg_times={args.reg_times} --moved_dir={moved_dir} --target=\"nvidia/nvidia-a100\" --candidate_cache_dir={to_measure_dir_file} --result_cache_dir={measured_tmp_file} > run_{part_id}.log 2>&1"
                returncode = exec_cmd_if_error_send_mail(command)
                if(returncode != 0):
                    time.sleep(3)
                    returncode = exec_cmd_if_error_send_mail(command)
                else:
                    break
            # to_measure_bak_file = os.path.join("measure_data/to_measure_bak", os.path.basename(to_measure_file))
            command = f"rm -rf {to_measure_dir_file}"
            exec_cmd_if_error_send_mail(command)

            measured_file = os.path.join("measure_data/measured_part", os.path.basename(to_measure_file))
            with lock:
                command = f"mv {measured_tmp_file} {measured_file}"
            exec_cmd_if_error_send_mail(command)
            continue

        print(f"{gpu_id} sleep...")
        time.sleep(10)


os.makedirs('measure_data/to_measure', exist_ok=True)
os.makedirs('measure_data/moved', exist_ok=True)
os.makedirs('measure_data/measured', exist_ok=True)
os.makedirs('measure_data/measured_part', exist_ok=True)
os.makedirs('measure_data/measured_tmp', exist_ok=True)
os.makedirs('measure_data/to_measure_bak', exist_ok=True)

os.system(f'mv measure_data/to_measure_*/*_part_* measure_data/to_measure/')

try:
    # 注意，我们再measure_programs中指定了target=a6000，这里指定的target不会生效，只会和文件夹有关。

    # 下面的进程都会启动，并同时运行，只有到了p.join时才会阻塞主线程
    if(args.cuda_id1 == "-1"):
        available_ids = [args.cuda_id2]
    elif(args.cuda_id2 == "-1"):
        available_ids = [args.cuda_id1]
    else:
        available_ids = [args.cuda_id1,args.cuda_id2]
    processes = []

    lock = Lock()

    p = Process(target=divide_worker, args=(lock, ))
    p.start()
    processes.append(p)
    time.sleep(1)
    p = Process(target=merge_worker, args=(lock, ))
    p.start()
    processes.append(p)
#
    for id in available_ids:
        p = Process(target=worker, args=(id, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
#创建多个进程来执行不同的任务，其中包括 divide_worker 和 merge_worker 函数，以及针对每个 available_ids 中的 ID 调用的 worker 函数。
#通过使用锁来管理对共享资源的访问，确保线程安全。
#启动所有进程，并在主进程中等待所有子进程完成。
except:
    print("Received KeyboardInterrupt, terminating workers")
    for p in processes:
        p.terminate()

# PYTHONUNBUFFERED=1 python run_measure.py --reg_times=1.5 --cuda_id1=MIG-1225b991-2e5a-522d-bdc9-c524f5000c68 --cuda_id2=MIG-4aacf663-1ed4-5fd6-960a-4009ed0e384e |& tee run.log
# PYTHONUNBUFFERED=1 python run_measure.py --reg_times=-1 --cuda_id1=MIG-ceaa2e03-b413-597f-8c77-86b869b62981  |& tee run.log
# PYTHONUNBUFFERED=1 python run_measure.py --reg_times=-1 --cuda_id1=MIG-0c650194-a353-5838-bf84-e51155ede733 --cuda_id2=MIG-ceaa2e03-b413-597f-8c77-86b869b62981 |& tee run.log
