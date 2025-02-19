#!/bin/bash

# 脚本名称: rename_by_flags.sh
# 功能: 根据布尔参数 a 和 b 修改目标文件名
# 用法: ./rename_by_flags.sh [a:true|false] [b:default|equal|preferl1|prefershared]
# 输入参数校验
if [ $# -ne 2 ]; then
    echo "错误：参数数量不正确"
    echo "用法: $0 <a:true|false> <b:default|equal|preferl1|prefershared>"
    exit 1
fi

a="$1"
b="$2"
target_file_reg="../src/target/source/codegen_cuda.cc"
target_file_reg_common="cppfile/codegen_cuda_common.cc.bak"
target_file_reg_maxnreg="cppfile/codegen_cuda_maxnreg.cc.bak"

target_file_sharemem="../src/runtime/cuda/cuda_module.cc"
target_file_sharemem_default="cppfile/cuda_module_default.cc.bak"
target_file_sharemem_equal="cppfile/cuda_module_equal.cc.bak"
target_file_sharemem_preferl1="cppfile/cuda_module_preferl1.cc.bak"
target_file_sharemem_prefershared="cppfile/cuda_module_prefershared.cc.bak"


# 核心重命名逻辑
if [ "$a" = "true" ]; then
    target_file_reg_new_name=$target_file_reg_maxnreg
elif [ "$a" = "false" ]; then
    target_file_reg_new_name=$target_file_reg_common
else
    echo "提示：a 必须为true或false"
    exit 0
fi

if [ "$b" = "default" ]; then
    target_file_shared_men_new_name=$target_file_sharemem_default
elif [ "$b" = "preferl1" ]; then
    target_file_shared_men_new_name=$target_file_sharemem_preferl1
elif [ "$b" = "prefershared" ]; then
    target_file_shared_men_new_name=$target_file_sharemem_prefershared
elif [ "$b" = "preferequal" ]; then
    target_file_shared_men_new_name=$target_file_sharemem_equal
else
    echo "提示 b 需为 default、preferl1、prefershared、preferequal"
    exit 0
fi

rm "$target_file_reg"
if [ $? -eq 0 ]; then
    echo "rm $target_file_reg"
else
    echo "$?" >&2
fi

rm "$target_file_sharemem"
if [ $? -eq 0 ]; then
    echo "rm $target_file_sharemem"
else
    echo "$?" >&2
fi

cp "$target_file_shared_men_new_name" "$target_file_sharemem"
if [ $? -eq 0 ]; then
    echo "cp $target_file_shared_men_new_name $target_file_sharemem"
else
    echo "$?" >&2
fi


cp "$target_file_reg_new_name" "$target_file_reg"
if [ $? -eq 0 ]; then
    echo "cp $target_file_reg_new_name $target_file_reg"
else
    echo "$?" >&2
fi


cd "../build"
make -j 64
cd -
