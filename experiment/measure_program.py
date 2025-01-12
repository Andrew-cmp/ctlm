import argparse
import glob
import os
import tvm
from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
from tvm.target import Target
import logging
import shutil
#只针对单个task测量，并且将cuda代码保存到新的文件夹中
def _parse_args():
    parser = argparse.ArgumentParser()
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
    return parser.parse_args()
# from tvm.contrib import nvcc
# @tvm.register_func("tvm_callback_cuda_compile", override=True)
# def tvm_callback_cuda_compile(code):
#     ptx = nvcc.compile_cuda(code)
#     print("this is override tvm_callback_cuda_compile")
#     with open("test_shared_dyn.ptx",'wb')as f:
#         f.write(ptx)
#     return ptx

# pylint: disable=too-many-locals
def deal_with_candidates(candidates):
    # print(f"len(candidates):{len(candidates)}")
    # print("*"*30)
    # print(f"sch0:{candidates[0].sch}")
    # print("*"*30)
    # print(f"sch1:{candidates[1].sch}")
    # print("*"*30)
    # print(f"mod0:{candidates[0].sch.mod}")
    # print("*"*30)
    # print(f"mod1:{candidates[1].sch.mod}")
    # print("*"*30)
    
    func = candidates[1].sch.mod["main"]
    func_with_attr = func.with_attr({"some_attr": "attr_value"})
    candidates[1].sch.mod.update_func(candidates[1].sch.mod.get_global_var("main"), func_with_attr)
    with open("1.tmp","w") as f:
        f.write(str(candidates[1].sch.mod))
    input("continue...")

def measure_candidates(database, builder, runner, task_record,new_dir):
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
    deal_with_candidates(candidates)
    with ms.Profiler() as profiler:
        for idx in range(0, len(candidates), args.batch_size):
            batch_candidates = candidates[idx : idx + args.batch_size]
            task_record._set_measure_candidates(batch_candidates)  # pylint: disable=protected-access
            with ms.Profiler.timeit("build"):
                task_record._send_to_builder(builder)  # pylint: disable=protected-access
            with ms.Profiler.timeit("run"):
                task_record._send_to_runner(runner)  # pylint: disable=protected-access
                batch_runner_results = task_record._join()  # pylint: disable=protected-access
            runner_results.extend(batch_runner_results)
            for i, result in enumerate(task_record.builder_results):
                if result.error_msg is None:
                    #print(result.artifact_path)
                    # artifact_dir = os.path.dirname(result.artifact_path)
                    # src = os.path.join(artifact_dir,"cuda_code.cu")
                    # dst_dir = os.path.join(new_dir,"cuda_code")
                    # dst = os.path.join(dst_dir,f"{i+idx}.cu")
                    # shutil.move(src,dst)
                    ms.utils.remove_build_dir(result.artifact_path)
                    
                else:
                    build_fail_indices.append(i + idx)
            task_record._clear_measure_state(batch_runner_results)  # pylint: disable=protected-access

    model_name, workload_name = database.path_workload.split("/")[-2:]
    record_name = database.path_tuning_record.split("/")[-1]
    new_database = ms.database.JSONDatabase(
        path_workload=os.path.join(args.result_cache_dir, model_name, workload_name),
        path_tuning_record=os.path.join(args.result_cache_dir, model_name, record_name),
    )
    workload = tuning_records[0].workload
    new_database.commit_workload(workload.mod)
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
        else:
            run_fail_indices.append(i)
    fail_indices_name = workload_name.replace("_workload.json", "_failed_indices.txt")
    with open(
        os.path.join(args.result_cache_dir, model_name, fail_indices_name), "w", encoding="utf8"
    ) as file:
        file.write(" ".join([str(n) for n in run_fail_indices]))
    print(
        f"Builder time: {profiler.get()['build']}, Runner time: {profiler.get()['run']}\n\
            Failed number of builds: {len(build_fail_indices)},\
            Failed number of runs: {len(run_fail_indices)}"
    )


args = _parse_args()  # pylint: disable=invalid-name


def main():
    logging.basicConfig(level=logging.INFO)
    

    builder = ms.builder.LocalBuilder(timeout_sec=300)
    runner = ms.runner.LocalRunner(timeout_sec=100)
    model_name = args.candidate_cache_dir.split("/")[-1]
    if not os.path.isdir(args.candidate_cache_dir):
        raise Exception("Please provide a correct candidate cache dir.")
    new_dir = os.path.join(args.result_cache_dir, model_name)
    try:
        os.makedirs(new_dir, exist_ok=True)
    except OSError:
        print(f"Directory {args.result_cache_dir} cannot be created successfully.")
    new_cuda_dir = os.path.join(new_dir, "cuda_code")
    try:
        os.makedirs(new_cuda_dir, exist_ok=True)
    except OSError:
        print(f"Directory {new_cuda_dir} cannot be created successfully.")
    
    new_ptx_dir = os.path.join(new_dir, "ptx_code")
    try:
        os.makedirs(new_ptx_dir, exist_ok=True)
    except OSError:
        print(f"Directory {new_ptx_dir} cannot be created successfully.")
         
        
    task_record = ms.task_scheduler.task_scheduler.TaskRecord(
        ms.TuneContext(target=Target(args.target)))
    
    database = ms.database.JSONDatabase(work_dir=args.candidate_cache_dir)
    measure_candidates(database, builder, runner, task_record,new_dir)


if __name__ == "__main__":
    main()
#python measure_program.py --result_cache_dir=dataset/tmp --candidate_cache_dir=dataset/to_measure_programs/a6000/450115668279416192__fused_nn_conv2d_add --target=nvidia/nvidia-a100