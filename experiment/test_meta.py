import argparse
import logging
from distutils.util import strtobool
from typing import Optional,Tuple
import tvm
from tvm import meta_schedule as ms
from tvm import tir
from tvm.meta_schedule.testing.te_workload import create_te_workload
from tvm.support import describe
from tvm import te
from tvm import topi
def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--number",
        type=int,
        default=3,
    )
    args.add_argument(
        "--repeat",
        type=int,
        default=1,
    )
    args.add_argument(
        "--min-repeat-ms",
        type=int,
        default=100,
    )
    args.add_argument(
        "--adaptive-training",
        type=lambda x: bool(strtobool(x)),
        required=False,
        help="example: True / False",
        default=True,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    return parsed


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()

def matmul(
    n: int, m: int, k: int, in_dtype: str = "float32", out_dtype: str = "float32"
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    a = te.placeholder((n, k), name="A", dtype=in_dtype)
    b = te.placeholder((k, m), name="B", dtype=in_dtype)
    k = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (n, m),
        lambda i, j: te.sum(a[i, k].astype(out_dtype) * b[k, j].astype(out_dtype), axis=[k]),
        name="C",
    )
    return (a, b, c)


def matmul_relu(
    n: int, m: int, k: int, in_dtype: str = "float32", out_dtype: str = "float32"
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    a = te.placeholder((n, k), name="A", dtype=in_dtype)
    b = te.placeholder((k, m), name="B", dtype=in_dtype)
    k = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (n, m),
        lambda i, j: te.sum(a[i, k].astype(out_dtype) * b[k, j].astype(out_dtype), axis=[k]),
        name="C",
    )
    d = topi.nn.relu(c)  # pylint: disable=invalid-name
    return (a, b, d)


def main():
    describe()
    print(f"Workload: {ARGS.workload}")
    mod=create_te_workload(ARGS.workload, 0)
    with ms.Profiler() as profiler:
        database= ms.tir_integration.tune_tir(
            mod=mod,
            target=ARGS.target,
            work_dir=ARGS.work_dir,
            max_trials_global=ARGS.num_trials,
            num_trials_per_iter=64,
            cost_model=ms.cost_model.XGBModel(  # type: ignore
                extractor=ms.feature_extractor.PerStoreFeature(),
                adaptive_training=ARGS.adaptive_training,
            ),
            strategy=ms.search_strategy.EvolutionarySearch(),
            task_name=ARGS.workload,
        )

    print("Tuning Time:")
    print(profiler.table())
    sch = ms.tir_integration.compile_tir(database, mod, ARGS.target)
    if sch is None:
        print("No valid schedule found!")
    else:
        print(sch.mod.script())
        print(sch.trace)


if __name__ == "__main__":
    main()
#python test_meta.py --workload=C2D --target=nvidia/nvidia-a6000 --work-dir=./experment --num-trials=100