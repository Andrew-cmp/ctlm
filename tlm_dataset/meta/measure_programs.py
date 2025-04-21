import argparse
import glob
import os
import sys
from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
from tvm.target import Target
import logging
import shutil
import tvm
import json
import math
from tvm.meta_schedule.utils import cpu_count

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_error_threshold", type=int, help="this vaule is related to mps tool.",default=1000
    )
    parser.add_argument(
        "--candidate_cache_dir", type=str, help="Please provide the full path to the candidates."
    )
    parser.add_argument(
        "--moved_dir", type=str, help="Please provide the full path that will store the failed tasks."
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
        "--rpc_host", type=str, help="Please provide the private IPv4 address for the tracker."
    )
    parser.add_argument(
        "--rpc_port", type=int, default=4445, help="Please provide the port for the tracker."
    )
    parser.add_argument(
        "--rpc_key",
        type=str,
        default="p3.2xlarge",
        help="Please provide the key for the rpc servers.",
    )
    parser.add_argument(
        "--builder_timeout_sec",
        type=int,
        default=10,
        help="The time for the builder session to time out.",
    )
    parser.add_argument(
        "--min_repeat_ms", type=int, default=100, help="The time for preheating the gpu."
    )
    parser.add_argument(
        "--runner_timeout_sec",
        type=int,
        default=100,
        help="The time for the runner session to time out.",
    )
    parser.add_argument(
        "--cpu_flush", type=bool, default=False, help="Whether to enable cpu cache flush or not."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size of candidates sent to builder and runner each time.",
    )
    parser.add_argument(
        "--cores",
        type=str,
        choices = ["physical","logical"],
        default="physical",
        help="",
    )
    parser.add_argument(
        "--reg_times",
        type=float,
        default=-1,
        help="the value that limit usage of reg when start thread",
    )
    return parser.parse_args()
class MPSError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
def rm_dir(dirs,path):
    for dir in dirs:
        tmp = os.path.join(path,dir)
        shutil.rmtree(tmp)
# @tvm._ffi.register_func("tvm_codegen_maxreg",override=True)
# def tvm_codegen_maxreg():
#     # with open("1.json",'w') as f:
#     #     json.load("1.json")
#     i = -1
#     return i
def add_candidates_func_attr(candidates,model_name):
    file_name = model_name
    register_path = "/home/hwhu/ctlm/ctlm/dataset/measure_register/measured/a100_100_100_100"
    register_path = os.path.join(register_path,file_name)
    register_json_path = os.path.join(register_path,"register.json")
    with open(register_json_path,"r") as f:
        registers = json.load(f)
    registers_dict = dict()
    for register in registers:
        registers_dict.update(register)
    avg = math.ceil(sum(registers_dict.values())/len(registers_dict))
    #print(registers)
    #print(registers_dict)
    for i, candidate in enumerate(candidates):
        func = candidate.sch.mod["main"]
        name = f"{i}.cu"
        register = registers_dict.get(name,avg)
        register_limitation = math.ceil(register/args.reg_times)
        func_with_attr = func.with_attr({"register": register_limitation})
        candidate.sch.mod.update_func(candidate.sch.mod.get_global_var("main"), func_with_attr)
        #input("continue...")
    return candidates

def measure_candidates(database, builder, runner, task_record):
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
        for idx in range(0, len(candidates), args.batch_size):
            batch_candidates = candidates[idx : idx + args.batch_size]
            task_record._set_measure_candidates(batch_candidates)  # pylint: disable=protected-access
            with ms.Profiler.timeit("build"):
                task_record._send_to_builder(builder)  # pylint: disable=protected-access
                # @tvm.register_func("tvm_codegen_maxreg",override=True)
                # def tvm_codegen_maxreg():
                #     i = 100000
                #     print("Yes")
                #     return i
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
        path_workload=os.path.join(args.result_cache_dir, model_name, workload_name),
        path_tuning_record=os.path.join(args.result_cache_dir, model_name, record_name),
    )
    workload = tuning_records[0].workload
    new_database.commit_workload(workload.mod)
    result_error_threshold = args.result_error_threshold
    result_error_num = 0
    result_error_flag = 0
    for i, (record, result) in enumerate(zip(tuning_records, runner_results)):
        if result.error_msg is None:
            new_database.commit_tuning_record(
                ms.database.TuningRecord(
                    trace=record.trace,
                    workload=workload,
                    run_secs=[v.value for v in result.run_secs],
                    target=Target(args.target),
                )
            )
            result_error_flag = 0
            result_error_num = 0
        else:
            run_fail_indices.append(i)
            if result_error_flag == 1:
                result_error_num += 1
            elif result_error_flag == 0:
                result_error_num = 1
                result_error_flag = 1
        if result_error_num >= result_error_threshold:
            raise MPSError("error")

    fail_indices_name = workload_name.replace("_workload.json", "_failed_indices.txt")
    build_fail_indices_name = workload_name.replace("_workload.json", "_build_failed_indices.txt")
    with open(
        os.path.join(args.result_cache_dir, model_name, fail_indices_name), "w", encoding="utf8"
    ) as file:
        file.write(" ".join([str(n) for n in run_fail_indices]))
    with open(
        os.path.join(args.result_cache_dir, model_name, build_fail_indices_name), "w", encoding="utf8"
    ) as file:
        file.write(" ".join([str(n) for n in build_fail_indices]))
    logging.info(
        f"Builder time: {profiler.get()['build']}, Runner time: {profiler.get()['run']}\n\
            Build model is {model_name}\n\
            Failed number of builds: {len(build_fail_indices)},\
            Failed number of runs: {len(run_fail_indices)}"
    )


args = _parse_args()  # pylint: disable=invalid-name


def main():
    logging.basicConfig(level=logging.INFO)
    if(args.cores == "physical"):
        cpu_cores = cpu_count(False)
    else:
        cpu_cores = cpu_count(True)
        
    builder = ms.builder.LocalBuilder(timeout_sec=3000,max_workers=cpu_cores)
    runner = ms.runner.LocalRunner(timeout_sec=100)
    if not os.path.isdir(args.candidate_cache_dir):
        raise Exception("Please provide a correct candidate cache dir.")
    try:
        os.makedirs(args.result_cache_dir, exist_ok=True)
    except OSError:
        print(f"Directory {args.result_cache_dir} cannot be created successfully.")
    # 这行代码的作用是查找指定目录下的所有子目录或文件，并将其路径存储到 model_dirs 变量中。
    model_dirs = glob.glob(os.path.join(args.candidate_cache_dir, "*"))

    task_record = ms.task_scheduler.task_scheduler.TaskRecord(
        ms.TuneContext(target=Target(args.target)))


    model_handled_dirs = []
    for model_dir in tqdm(model_dirs):
        # such as '14729483509063283358__fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu'
        model_name = model_dir.split("/")[-1]
        new_dir = os.path.join(args.result_cache_dir, model_name)
        if os.path.isdir(new_dir):
            recods_path = os.path.join(new_dir, 'database_tuning_record.json')
            if os.path.exists(recods_path):
                with open(recods_path, 'r') as f:
                    lines = [line for line in f.read().strip().split('\n') if line]
                    if len(lines) > 0:
                        continue
        else:
            os.makedirs(new_dir)
        database = ms.database.JSONDatabase(work_dir=model_dir)
        try:
            measure_candidates(database, builder, runner, task_record)
        except MPSError as e:
            try:
                rm_dir([model_name],args.result_cache_dir)             #移除在result_cache_dir中已经创建了的task的dir
                rm_dir(model_handled_dirs,args.candidate_cache_dir)    #移除candidate_cache_dir中已经测量过的task
            except FileNotFoundError:
                print(model_handled_dirs)
                print(f"model name is {model_name}")
            print(f"Failed task is {model_name}")
            source_dir = os.path.join(args.candidate_cache_dir,model_name)
            destination_dir = os.path.join(args.moved_dir,model_name)
            shutil.move(source_dir, destination_dir)
            print(f"Failed task is removed to moved dir")
            sys.exit(1)
        else:
            model_handled_dirs.append(model_name)


if __name__ == "__main__":
    print(args.candidate_cache_dir)
    main()
# python measure_programs.py \
# --result_cache_dir=dataset/tmp \
# --candidate_cache_dir=/home/hwhu/ctlm/ctlm/dataset/to_measure_programs/v100 \
# --target=nvidia/nvidia-a100 \
# --reg_times=2 \
# --result_error_threshold=5 \
# --moved_dir=dataset/tmp


# python measure_programs.py \
# --result_cache_dir=ctlm_data/ctlm_record_for_eval/gen_eval_response.json_measured \
# --candidate_cache_dir=ctlm_data/ctlm_record_for_eval/gen_eval_response.json \
# --target=nvidia/nvidia-a6000 \
# --reg_times=-1 \
# --result_error_threshold=5 \
# --moved_dir=dataset/tmp \
# >run.log 2>&1
