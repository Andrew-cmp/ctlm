from tvm import tir
from common import save_sch_mod, build_check_evaluate
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  b1 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  save_sch_mod(sch)
  b2 = sch.reindex(block=b0, buffer=("write", 0))
  save_sch_mod(sch)
  b3 = sch.reindex(block=b0, buffer=("read", 0))
  b4 = sch.reindex(block=b0, buffer=("read", 1))
  save_sch_mod(sch)
  sch.transform_layout(block=b0, buffer=("read", 0), index_map=lambda vi, vk: (vi, vk,), pad_value=None, assume_injective_transform=True)
  sch.transform_layout(block=b0, buffer=("read", 1), index_map=lambda vj, vk: (vk, vj,), pad_value=None, assume_injective_transform=True)
  sch.transform_layout(block=b0, buffer=("write", 0), index_map=lambda vi, vj: (vi, vj,), pad_value=None, assume_injective_transform=True)
  save_sch_mod(sch)
  sch.transform_block_layout(block=b2, index_map=lambda vi, vj: (vi, vj,))
  sch.transform_block_layout(block=b3, index_map=lambda vi, vk: (vi, vk,))
  sch.transform_block_layout(block=b4, index_map=lambda vj, vk: (vk, vj,))
  sch.transform_block_layout(block=b0, index_map=lambda vi, vj, vk: (vi, vj, vk,))
  save_sch_mod(sch)
  l5, l6, l7 = sch.get_loops(block=b0)
  save_sch_mod(sch)
  l8, l9 = sch.split(loop=l7, factors=[None, 16], preserve_unit_iters=True)
  save_sch_mod(sch)
  l10, l11 = sch.split(loop=l6, factors=[None, 32], preserve_unit_iters=True)
  save_sch_mod(sch)
  l12, l13 = sch.split(loop=l5, factors=[None, 8], preserve_unit_iters=True)
  save_sch_mod(sch)
  l14, l15, l16, l17, l18, l19 = sch.get_loops(block=b0)
  sch.reorder(l16, l18, l13, l11, l9)
  save_sch_mod(sch)
  b20 = sch.blockize(target=l13, preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.annotate(block_or_loop=b20, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_8x32x16_f16f16f16")
  save_sch_mod(sch)
  sch.annotate(block_or_loop=b20, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_8x32x16_f16")
  sch.annotate(block_or_loop=b20, ann_key="warp_execution", ann_val=1)
  save_sch_mod(sch)
  l21, l22, l23 = sch.get_loops(block=b20)
  v24, v25, v26, v27, v28 = sch.sample_perfect_tile(loop=l21, n=5, max_innermost_factor=4, decision=[1, 8, 1, 1, 1])
  l29, l30, l31, l32, l33 = sch.split(loop=l21, factors=[v24, v25, v26, v27, v28], preserve_unit_iters=True)
  save_sch_mod(sch)
  v34, v35, v36, v37, v38 = sch.sample_perfect_tile(loop=l22, n=5, max_innermost_factor=4, decision=[2, 2, 1, 1, 2])
  l39, l40, l41, l42, l43 = sch.split(loop=l22, factors=[v34, v35, v36, v37, v38], preserve_unit_iters=True)
  save_sch_mod(sch)
  v44, v45, v46 = sch.sample_perfect_tile(loop=l23, n=3, max_innermost_factor=4, decision=[1, 8, 1])
  l47, l48, l49 = sch.split(loop=l23, factors=[v44, v45, v46], 
                            preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.reorder(l29, l39, l30, l40, l31, l41, l47, l48, l32, l42, l49, l33, l43)
  save_sch_mod(sch)
  l50 = sch.fuse(l29, l39, preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.bind(loop=l50, thread_axis="blockIdx.y")
  save_sch_mod(sch)
  l51 = sch.fuse(l30, l40, preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.bind(loop=l51, thread_axis="blockIdx.x")
  save_sch_mod(sch)
  l52 = sch.fuse(l31, l41, preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.bind(loop=l52, thread_axis="threadIdx.y")
  save_sch_mod(sch)
  sch.annotate(block_or_loop=b20, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=1)
  sch.annotate(block_or_loop=b20, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
  save_sch_mod(sch)
  sch.transform_layout(block=b20, buffer=("write", 0), index_map=lambda i0, i1: (i0 // 8 // (v27 * v28), i1 // 32 // (v37 * v38), i0 // 8 % (v27 * v28), i1 // 32 % (v37 * v38), i0 % 8, i1 % 32,), pad_value=None, assume_injective_transform=True)
  save_sch_mod(sch)
  b53 = sch.cache_write(block=b20, write_buffer_index=0, storage_scope="shared.dyn")
  save_sch_mod(sch)
  sch.reverse_compute_at(block=b53, loop=l51, preserve_unit_loops=True, index=-1)
  save_sch_mod(sch)
  b54 = sch.cache_write(block=b20, write_buffer_index=0, storage_scope="wmma.accumulator")
  save_sch_mod(sch)
  l55, l56, l57, l58, l59, l60, l61, l62 = sch.get_loops(block=b53)
  sch.reorder(l59, l57, l58, l60)
  save_sch_mod(sch)
  sch.compute_at(block=b54, loop=l59, preserve_unit_loops=True, index=-1)
  save_sch_mod(sch)
  l63, l64, l65, l66, l67, l68, l69, l70, l71 = sch.get_loops(block=b54)
  l72 = sch.fuse(l66, l67, preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.bind(loop=l72, thread_axis="threadIdx.y")
  save_sch_mod(sch)
  sch.reverse_compute_inline(block=b2)
  save_sch_mod(sch)
  l73, l74, l75, l76, l77, l78, l79, l80 = sch.get_loops(block=b54)
  b81 = sch.blockize(target=l79, preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.annotate(block_or_loop=b81, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_8x32x16_f16_shared_dyn")
  save_sch_mod(sch)
  l82, l83, l84, l85, l86, l87, l88, l89 = sch.get_loops(block=b53)
  l90 = sch.fuse(l85, l86, l87, l88, l89, preserve_unit_iters=True)
  save_sch_mod(sch)
  v91 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
  sch.annotate(block_or_loop=b53, ann_key="meta_schedule.cooperative_fetch", ann_val=v91)
  save_sch_mod(sch)
  b92 = sch.cache_read(block=b20, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b20])
  save_sch_mod(sch)
  sch.compute_at(block=b92, loop=l47, preserve_unit_loops=True, index=-1)
  save_sch_mod(sch)
  l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b92)
  l99 = sch.fuse(l97, l98, preserve_unit_iters=True)
  save_sch_mod(sch)
  v100 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b92, ann_key="meta_schedule.cooperative_fetch", ann_val=v100)
  save_sch_mod(sch)
  b101 = sch.cache_read(block=b20, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b20])
  save_sch_mod(sch)
  sch.compute_at(block=b101, loop=l47, preserve_unit_loops=True, index=-1)
  save_sch_mod(sch)
  l102, l103, l104, l105, l106, l107 = sch.get_loops(block=b101)
  l108 = sch.fuse(l106, l107, preserve_unit_iters=True)
  save_sch_mod(sch)
  v109 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=2)
  sch.annotate(block_or_loop=b101, ann_key="meta_schedule.cooperative_fetch", ann_val=v109)
  save_sch_mod(sch)
  b110 = sch.cache_read(block=b20, read_buffer_index=0, storage_scope="wmma.matrix_a")
  save_sch_mod(sch)
  sch.compute_at(block=b110, loop=l48, preserve_unit_loops=True, index=-1)
  save_sch_mod(sch)
  l111, l112, l113, l114, l115, l116, l117 = sch.get_loops(block=b110)
  l118, l119 = sch.split(loop=l117, factors=[None, 16], preserve_unit_iters=True)
  l120, l121 = sch.split(loop=l116, factors=[None, 8], preserve_unit_iters=True)
  save_sch_mod(sch)
  l122, l123, l124, l125, l126, l127, l128, l129, l130 = sch.get_loops(block=b110)
  sch.reorder(l129, l121, l119)
  save_sch_mod(sch)
  b131 = sch.blockize(target=l121, preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.annotate(block_or_loop=b131, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_8x32x16_f16_a_shared_dyn")
  save_sch_mod(sch)
  b132 = sch.cache_read(block=b20, read_buffer_index=1, storage_scope="wmma.matrix_b")
  save_sch_mod(sch)
  sch.compute_at(block=b132, loop=l48, preserve_unit_loops=True, index=-1)
  save_sch_mod(sch)
  l133, l134, l135, l136, l137, l138, l139 = sch.get_loops(block=b132)
  l140, l141 = sch.split(loop=l139, factors=[None, 32], preserve_unit_iters=True)
  l142, l143 = sch.split(loop=l138, factors=[None, 16], preserve_unit_iters=True)
  l144, l145, l146, l147, l148, l149, l150, l151, l152 = sch.get_loops(block=b132)
  sch.reorder(l151, l143, l141)
  save_sch_mod(sch)
  b153 = sch.blockize(target=l143, preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.annotate(block_or_loop=b153, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_8x32x16_f16_b_shared_dyn")
  save_sch_mod(sch)
  b154, = sch.get_producers(block=b92)
  sch.compute_inline(block=b154)
  save_sch_mod(sch)
  sch.storage_align(block=b92, buffer_index=0, axis=-2, factor=32, offset=8)
  save_sch_mod(sch)
  b155, = sch.get_producers(block=b101)
  sch.compute_inline(block=b155)
  save_sch_mod(sch)
  sch.storage_align(block=b101, buffer_index=0, axis=-2, factor=32, offset=8)
  save_sch_mod(sch)
  v156 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=4)
  sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v156)
  save_sch_mod(sch)
  sch.enter_postproc()
  save_sch_mod(sch)
  sch.unannotate(block_or_loop=b53, ann_key="meta_schedule.cooperative_fetch")
  save_sch_mod(sch)
  l157, l158, l159, l160 = sch.get_loops(block=b53)
  l161, l162, l163, l164 = sch.split(loop=l160, factors=[None, 1, 32, 8], preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.vectorize(loop=l164)
  save_sch_mod(sch)
  sch.bind(loop=l163, thread_axis="threadIdx.x")
  sch.bind(loop=l162, thread_axis="threadIdx.y")
  save_sch_mod(sch)
  sch.unannotate(block_or_loop=b92, ann_key="meta_schedule.cooperative_fetch")
  save_sch_mod(sch)
  l165, l166, l167, l168, l169 = sch.get_loops(block=b92)
  l170, l171, l172, l173 = sch.split(loop=l169, factors=[None, 1, 32, 2], preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.vectorize(loop=l173)
  sch.bind(loop=l172, thread_axis="threadIdx.x")
  sch.bind(loop=l171, thread_axis="threadIdx.y")
  save_sch_mod(sch)
  sch.unannotate(block_or_loop=b101, ann_key="meta_schedule.cooperative_fetch")
  save_sch_mod(sch)
  l174, l175, l176, l177, l178 = sch.get_loops(block=b101)
  l179, l180, l181, l182 = sch.split(loop=l178, factors=[None, 1, 32, 4], preserve_unit_iters=True)
  save_sch_mod(sch)
  sch.vectorize(loop=l182)
  sch.bind(loop=l181, thread_axis="threadIdx.x")
  sch.bind(loop=l180, thread_axis="threadIdx.y")
  save_sch_mod(sch)
  b183 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b183, ann_key="meta_schedule.unroll_explicit")
  save_sch_mod(sch)
  b184, b185, b186, b187, b188, b189, b190 = sch.get_child_blocks(b183)
  l191, l192, l193, l194, l195, l196, l197, l198 = sch.get_loops(block=b184)
  l199, l200, l201, l202, l203, l204, l205, l206 = sch.get_loops(block=b185)
  l207, l208, l209, l210, l211, l212, l213 = sch.get_loops(block=b186)
  l214, l215, l216, l217, l218, l219, l220 = sch.get_loops(block=b187)
  l221, l222, l223, l224, l225, l226, l227, l228, l229, l230 = sch.get_loops(block=b188)
  sch.annotate(block_or_loop=l221, ann_key="pragma_auto_unroll_max_step", ann_val=1024)
  sch.annotate(block_or_loop=l221, ann_key="pragma_unroll_explicit", ann_val=1)
  save_sch_mod(sch)
  l231, l232, l233, l234, l235, l236 = sch.get_loops(block=b189)
  l237, l238, l239, l240, l241, l242, l243 = sch.get_loops(block=b190)
  b244 = sch.get_block(name="C_o", func_name="main")
  l245, l246, l247, l248, l249, l250, l251, l252, l253, l254 = sch.get_loops(block=b244)
  b255 = sch.decompose_reduction(block=b244, loop=l248)
  save_sch_mod(sch)
  sch.unannotate(block_or_loop=b255, ann_key="meta_schedule.auto_tensorize")
  save_sch_mod(sch)
  sch.annotate(block_or_loop=b255, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_8x32x16_f16")
  save_sch_mod(sch)
  sch.unannotate(block_or_loop=b244, ann_key="meta_schedule.auto_tensorize_init")
  save_sch_mod(sch)
  sch.unannotate(block_or_loop=b255, ann_key="meta_schedule.auto_tensorize_init")
  save_sch_mod(sch)
  b256 = sch.get_block(name="C_o_init", func_name="main")
  sch.unannotate(block_or_loop=b256, ann_key="meta_schedule.auto_tensorize")
  save_sch_mod(sch)
  sch.tensorize(block_or_loop=b256, tensor_intrin="wmma_fill_8x32x16_f16", preserve_unit_iters=True)
  save_sch_mod(sch)
  b257 = sch.get_block(name="A_reindex_shared.dyn_wmma.matrix_a_o", func_name="main")
  sch.unannotate(block_or_loop=b257, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b257, tensor_intrin="wmma_load_8x32x16_f16_a_shared_dyn", preserve_unit_iters=True)
  save_sch_mod(sch)
  b258 = sch.get_block(name="B_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
  sch.unannotate(block_or_loop=b258, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b258, tensor_intrin="wmma_load_8x32x16_f16_b_shared_dyn", preserve_unit_iters=True)
  save_sch_mod(sch)
  b259 = sch.get_block(name="C_o_update", func_name="main")
  sch.unannotate(block_or_loop=b259, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b259, 
                tensor_intrin="wmma_sync_8x32x16_f16f16f16", 
                preserve_unit_iters=True)
  save_sch_mod(sch)
  b260 = sch.get_block(name="C_reindex_shared.dyn_wmma.accumulator_o", func_name="main")
  sch.unannotate(block_or_loop=b260, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b260, tensor_intrin="wmma_store_8x32x16_f16_shared_dyn", preserve_unit_iters=True)
  save_sch_mod(sch)