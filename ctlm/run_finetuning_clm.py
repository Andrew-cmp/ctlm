import subprocess, os

# 设置 screen 命令和相关参数
session_name = os.path.basename(os.path.abspath(__file__))
log_file = f'{session_name}.log'
session_name = session_name.replace('.', '_')

if os.path.exists(log_file):
    # tag = input(log_file + ' exist, delete it? [n]')
    # if tag == 'y':
        # 删除文件
    os.remove(log_file)

# 构建完整的 screen 命令
cmd = """tmux new -s %s -d '{
{
set -x
echo "#################################################################"
date

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=7,6,5,4 python train_clm.py \
                                    --do_train \
                                    --model_type=gpt2 \
                                    --tokenizer_name=ctlm_data/ctlm_tokenizer \
                                    --output_dir=ctlm_data/clm_gen_adapter_adapter \
                                    --dataset_name=ctlm_data/v100_gen_best \
                                    --per_device_train_batch_size=4 \
                                    \
                                    --overwrite_output_dir=True \
                                    --logging_steps=10 \
                                    --num_train_epochs=10 \
                                    --remove_unused_columns=False \
                                    --learning_rate=5e-5 \
                                    --model_name_or_path=ctlm_data/clm_gen \
                                    --lr_scheduler_type=constant \
                                    --start_finetuning \
                                    --adapter_name=bottleneck \

                                    # 4层是64,128
                                    # --save_strategy=epoch \


date
} |& tee -a %s
}'
""" % (session_name, log_file)

# 使用 subprocess 运行命令
subprocess.Popen(cmd, shell=True)
