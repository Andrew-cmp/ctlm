import tvm
import os
from tvm import meta_schedule as ms
from typing import List, Callable
from typing import Tuple, Union
from typing_extensions import Literal
import numpy as np
import tvm.testing
from tvm import relay
from tvm.meta_schedule.space_generator import PostOrderApply
import tvm.tir.tensor_intrin
from tvm.meta_schedule.search_strategy import MeasureCandidate
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def save_sch_mod_cuda_code_interact(sch: tvm.tir.Schedule, 
                 *,
                 old_mod_file: str = '/home/hwhu/ctlm/tvm_app/trash.print/old_mod_file.py', 
                 new_mod_file: str = '/home/hwhu/ctlm/tvm_app/trash.print/new_mod_file.py',
                 ):
    
    if os.path.exists(old_mod_file):
        os.remove(old_mod_file)
    if os.path.exists(new_mod_file):
        os.rename(new_mod_file, old_mod_file)
    with open(new_mod_file, 'w') as f:
        f.write(str(sch.mod))
    
    old_cuda_file = os.path.join(os.path.dirname(old_mod_file), 'old_cuda_code.cu')
    new_cuda_file = os.path.join(os.path.dirname(new_mod_file), 'new_cuda_code.cu')
    if os.path.exists(old_cuda_file):
        os.remove(old_cuda_file)
    if os.path.exists(new_cuda_file):
        os.rename(new_cuda_file, old_cuda_file)
    try:
        write_cuda_code(sch.mod, new_cuda_file)
    except:
        with open(new_cuda_file, 'w') as f:
            f.write("cuda emit failed")
    input(f"lastest trace uesd and cuda code write,  Press Enter to continue...")
def build_check_evaluate(sch: tvm.tir.Schedule, 
                    check_comp: Callable,
                    is_build: bool = True,
                    is_write_src: bool = True,
                    is_check: bool = False,
                    is_evaluate: bool = True,
                    target="nvidia/nvidia-a100",
                    func_name="main"):

    #if is_build:
    lib = tvm.build(sch.mod, target=target)
    print("build success ...")
    dev = tvm.cuda() if 'nvidia' in target else tvm.cpu()
    args_info:List[ms.arg_info.TensorInfo]  = ms.arg_info.ArgInfo.from_entry_func(sch.mod)
    args_info_json = [arg_info.as_json() for arg_info in args_info]
    input_nps = [np.random.random(aij[2]).astype(aij[1]) for aij in args_info_json]
    input_nds = [tvm.nd.array(inp, device=dev) for inp in input_nps]
    lib(*input_nds)
    print("run success")
        
    if is_write_src:
        with open('/home/houhw/tvm_learn/tvm_app/result_tir.h','w') as f:
            f.write(str(sch.mod))
        if "nvidia" in target:
            with open('/home/houhw/tvm_learn/tvm_app/result.h','w') as f:
                f.write(lib.imported_modules[0].get_source())
        else:
            with open('/home/houhw/tvm_learn/tvm_app/result.h','w') as f:
                f.write(lib.get_source("asm"))
        print("write src success ...")
    
    if is_check:
        np_res = check_comp(*input_nps[:-1]).astype("float64")
        tvm_res = input_nds[-1].numpy().astype("float64")
        check_res = np.mean(np.abs(np_res - tvm_res))
        print(f"check result: {check_res}")
    
    if is_evaluate:
        evaluator = lib.time_evaluator(func_name, dev, number=10)
        evaluator_time = evaluator(*input_nds).mean
        print(f"evaluator result: {evaluator_time}")
    return evaluator_time
    
def avx512_vnni_test_build(states: List[tvm.tir.Schedule], target):
    result = []
    builder = ms.builder.LocalBuilder(max_workers=8)
    batch_builder_inputs:List[ms.builder.BuilderInput] = []
    for state in states:
        batch_builder_inputs.append(ms.builder.BuilderInput(state.mod, tvm.target.Target(target)))
    builder_res:List[ms.builder.BuilderResult] = builder.build(batch_builder_inputs)
    err_cnt = 0
    for i, br in enumerate(builder_res):
        if br.error_msg is None:
            result.append(states[i])
        err_cnt += 1
    return result, err_cnt


def generate_candidates(mod, target,sch_rules = None, postprocs = None, mutator = None, number=1):
    if(postprocs !=None and mutator !=None and sch_rules != None ):
        generator = PostOrderApply(sch_rules=sch_rules, postprocs=postprocs, mutator_probs=mutator)
    else :
         generator = PostOrderApply()
    strategy = ms.search_strategy.EvolutionarySearch(init_measured_ratio=0.0,
                                                     genetic_num_iters=1,)
    sample_init_population = tvm.get_global_func(
        "meta_schedule.SearchStrategyEvolutionarySearchSampleInitPopulation"
    )
    evolve_with_cost_model = tvm.get_global_func(
        "meta_schedule.SearchStrategyEvolutionarySearchEvolveWithCostModel"
    )
    context = ms.TuneContext(
        mod=mod,
        target=tvm.target.Target(target),
        space_generator=generator,
        search_strategy=strategy,
    )
    design_space = context.generate_design_space()
    print(f"design space sketch: {len(design_space)}.")
    # design_space = design_space[1:2]
    context.pre_tuning(
            max_trials=7,
            design_spaces=design_space,
            database=ms.database.MemoryDatabase(),
            cost_model=ms.cost_model.RandomModel(),  # type: ignore
    )
    result: List[tvm.tir.Schedule] = []
    
    while len(result) == 0:
        states = sample_init_population(strategy, 100)
        result: List[tvm.tir.Schedule] = \
                    evolve_with_cost_model(strategy, states, int(len(states) * 1.1))
        if sch_rules == 'vnni' or sch_rules == 'avx512':
            result, err_cnt = avx512_vnni_test_build(result, target)

    schs = result[0:number]
    candidates = \
        [MeasureCandidate(sch, ms.arg_info.ArgInfo.from_entry_func(sch.mod)) for sch in schs]
    return candidates, schs

def get_MatmulModule(M, N, K, A_dtype, B_dtype, C_dtype, is_vnni_avx512=False, target=None):
    if not is_vnni_avx512:
        from tvm.script import tir as T
        @tvm.script.ir_module
        class MyMatmulModule:
            @T.prim_func
            def main(
                A: T.Buffer((M, K), A_dtype),
                B: T.Buffer((K, N), B_dtype),
                C: T.Buffer((M, N), C_dtype),
            ):
                T.func_attr({"global_symbol": "main", "tir.noalias": True})
                for i, j, k in T.grid(M, N, K):
                    with T.block("C"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        with T.init():
                            C[vi, vj] = 0.0
                        C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    if is_vnni_avx512:
        input_data = relay.var("input_data", shape=(M, K), dtype=A_dtype)
        weight_data = relay.var("weight", shape=(N, K), dtype=B_dtype)
        x = relay.nn.dense(input_data, weight_data, out_dtype=C_dtype)
        relay_mod = tvm.IRModule.from_expr(x)
        weight_np = np.random.uniform(1, 10, size=(N, K)).astype(B_dtype)
        params = {"weight": weight_np}
        extracted_tasks = ms.relay_integration.extract_tasks(relay_mod, target, params)
        assert len(extracted_tasks) == 1
        task = extracted_tasks[0]
        MyMatmulModule = task.dispatched[0]

    return MyMatmulModule

def get_ConvModule(input_shape, input_dtype, weight_shape, weight_dtype,
                   strides, padding, channels, kernel_size, input_layout, kernel_layout,
                   out_dtype, target):
        def fused_nn_conv2d(
            input_shape: Tuple[int, int, int, int],
            input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
            weight_shape: Tuple[int, int, int, int],
            weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
            strides: Tuple[int, int],
            padding: Tuple[int, int],
            channels: Tuple[int],
            kernel_size: Tuple[int, int],
            input_layout: Tuple[Literal["NCHW"], Literal["NHWC"]],
            kernel_layout: Tuple[Literal["OIHW"], Literal["HWIO"]],
            out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
        ):
            input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
            weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
            
            x = relay.nn.conv2d(data=input_data, weight=weight_data, strides=strides, padding=padding,
                                channels=channels, kernel_size=kernel_size, data_layout=input_layout,
                                kernel_layout=kernel_layout, out_dtype=out_dtype)
            x = relay.nn.relu(x)
            relay_mod = tvm.IRModule.from_expr(x)
            
            weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
            params = {"weight": weight_np}
            
            return relay_mod, params
        relay_mod, params = \
            fused_nn_conv2d(input_shape, input_dtype, weight_shape, weight_dtype,
                   strides, padding, channels, kernel_size, input_layout, kernel_layout,
                   out_dtype)
        extracted_tasks = ms.relay_integration.extract_tasks(relay_mod, target, params)
        for task in extracted_tasks:
            if "conv" in task.task_name or "batch_matmul" in task.task_name \
                or "dense" in task.task_name:
                    print(task.task_name)
                    return task.dispatched[0]

def get_Conv3dModule(input_shape, input_dtype, weight_shape, weight_dtype,
                     strides, padding, input_layout, kernel_layout, out_dtype,
                     kernel_size, channels,
                     target,
                   ):
    input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
    weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
    x = relay.nn.conv3d(data=input_data, weight=weight_data, strides=strides, padding=padding, data_layout=input_layout, kernel_layout=kernel_layout, out_dtype=out_dtype, kernel_size=kernel_size, channels=channels)
    relay_mod = tvm.IRModule.from_expr(x)
    weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
    params = {"weight": weight_np}
    extracted_tasks = ms.relay_integration.extract_tasks(relay_mod, target, params)
    return extracted_tasks[0].dispatched[0]

def get_BatchMatmul(input_shape, input_dtype, weight_shape, weight_dtype, out_dtype, target):
        def fused_nn_batch_matmul(
            input_shape: Tuple[int, int],
            input_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
            weight_shape: Tuple[int, int],
            weight_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
            out_dtype: Union[Literal["float16"], Literal["int8"], Literal["uint8"], Literal["int32"]],
        ):
            input_data = relay.var("input_data", shape=input_shape, dtype=input_dtype)
            weight_data = relay.var("weight", shape=weight_shape, dtype=weight_dtype)
            
            x = relay.nn.batch_matmul(input_data, weight_data, out_dtype=out_dtype, 
                                    transpose_b=True)
            
            relay_mod = tvm.IRModule.from_expr(x)
            weight_np = np.random.uniform(1, 10, size=weight_shape).astype(weight_dtype)
            params = {"weight": weight_np}
            
            return relay_mod, params
        relay_mod, params = \
            fused_nn_batch_matmul(input_shape, input_dtype, weight_shape, weight_dtype, out_dtype)
        extracted_tasks = ms.relay_integration.extract_tasks(relay_mod, target, params)
        for task in extracted_tasks:
            if "conv" in task.task_name or "batch_matmul" in task.task_name \
                or "dense" in task.task_name:
                    print(task.task_name)
                    return task.dispatched[0]
def write_cuda_code(mod,path=None):
    build_mod = tvm.build(mod, target="cuda")
    if path is not None:
        with open(path, 'w') as f:
            f.write(build_mod.imported_modules[0].get_source())
    else:
        with open('/home/houhw/tvm_learn/tvm_app/trash.print/cuda_code','w') as f:
            f.write(build_mod.imported_modules[0].get_source())   