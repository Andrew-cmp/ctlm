import tvm
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.relay_integration import extracted_tasks_to_tune_contexts, extract_tasks
from tvm.meta_schedule.logging import get_loggers_from_work_dir
import argparse
from common import register_data_path,load_tasks,get_task_hashes,remove_trailing_numbers
import argparse
import glob
import os

from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
from tvm.target import Target
import logging

from tqdm import tqdm  # type: ignore
from tvm import meta_schedule as ms
# pylint: disable=too-many-locals
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        default="nvidia/nvidia-v100",
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
        "--candidate_cache_dir", type=str, help="Please provide the full path to the candidates."
    )
    parser.add_argument(
        "--result_cache_dir", type=str, help="Please provide the full path to the result database."
    )
    return parser.parse_args()



def measure_candidates(database, builder, runner, task_record,path_task_result):
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
                    ms.utils.remove_build_dir(result.artifact_path)
                else:
                    build_fail_indices.append(i + idx)
            task_record._clear_measure_state(batch_runner_results)  # pylint: disable=protected-access

    model_name, workload_name = database.path_workload.split("/")[-2:]
    record_name = database.path_tuning_record.split("/")[-1]
    new_database = ms.database.JSONDatabase(
        path_workload=os.path.join(path_task_result, workload_name),
        path_tuning_record=os.path.join(path_task_result, record_name),
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
        os.path.join(path_task_result, fail_indices_name), "w", encoding="utf8"
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
    
    register_data_path(args.target)
    
    builder = ms.builder.LocalBuilder(timeout_sec=30)
    runner = ms.runner.LocalRunner(timeout_sec=10)
    args.candidate_cache_dir
    from common import TO_MEASURE_PROGRAM_FOLDER
    
    task_record = ms.task_scheduler.task_scheduler.TaskRecord(
        ms.TuneContext(target=Target(args.target)))
    
    path_tasks = glob.glob(os.path.join(TO_MEASURE_PROGRAM_FOLDER, "*"))
    ##tasks_dirs = os.listdir(TO_MEASURE_PROGRAM_FOLDER)
    ##两个返回值都是列表，区别是glob返回的路径下所有文件或文件夹的绝对路径，listdir是返回所有文件和文件夹的文件名。
    ## 当然glob还可以有更多用法。
    
    
    for path_task in tqdm(path_tasks):
        # such as '14729483509063283358__fused_nn_contrib_conv2d_winograd_without_weight_transform_add_add_nn_relu' 
        from common import MEASURE_RECORD_FOLDER
        path_task_result = os.path.join(MEASURE_RECORD_FOLDER,os.path.basename(path_task))
        if os.path.isdir(path_task_result):
            recods_path = os.path.join(path_task, 'database_tuning_record.json')
            if os.path.exists(recods_path):
                with open(recods_path, 'r') as f:
                    lines = [line for line in f.read().strip().split('\n') if line]
                    if len(lines) > 0:
                        continue
        else:
            os.makedirs(path_task_result)
        database = ms.database.JSONDatabase(work_dir=path_task)
        measure_candidates(database, builder,runner, task_record,path_task_result)


if __name__ == "__main__":
    main()