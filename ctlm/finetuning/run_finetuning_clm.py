import subprocess, os
import argparse
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=str, help="target which adapter will be trained on",default=1000
    )
    
    return parser.parse_args()
args = _parse_args()
model_list = ['v100', 'a100', '2080', 'None','a6000','4090','t4','a40','l20','l40','4090d','3090','xp','p4','p10','a4000']
for model in model_list:
    if model in args.target:
        break
assert(model != 'None')
# 设置 screen 命令和相关参数
session_name = os.path.basename(os.path.abspath(__file__))
log_file = f'{session_name}_{model}_train_adapters.log'
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
                                    # --output_dir=ctlm_data/clm_gen_adapter_adapter \
                                    # --dataset_name=ctlm_data/v100_gen_best \
                                    --per_device_train_batch_size=4 \
                                    \
                                    --overwrite_output_dir=True \
                                    --logging_steps=10 \
                                    --num_train_epochs=10 \
                                    --remove_unused_columns=False \
                                    --learning_rate=5e-5 \
                                    --model_name_or_path=ctlm_data/clm_gen \
                                    --lr_scheduler_type=constant \

                                    # 4层是64,128
                                    # --save_strategy=epoch \
                                        
                                    --dataset_name=ctlm_data/%s_gen_best \
                                    --output_dir=ctlm_data/clm_gen_best_adapter_%s \
                                    --train_adapter \
                                    --adapter_config seq_bn \
                                    




date
} |& tee -a %s
}'
""" % (session_name, model,model,log_file)

# 使用 subprocess 运行命令
subprocess.Popen(cmd, shell=True)
