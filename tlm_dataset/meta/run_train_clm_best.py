import subprocess, os
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from meta_common import register_data_path

@dataclass
class ScriptArguments:
    target: str = field(metadata={"help": ""})


parser = HfArgumentParser(ScriptArguments)
script_args: ScriptArguments = parser.parse_args_into_dataclasses()[0]
register_data_path(script_args.target)
from meta_common import CURRENT_DATASET_FOLDER
from meta_common import HARDWARE_PLATFORM

# 设置 screen 命令和相关参数
session_name = os.path.basename(os.path.abspath(__file__))+f"_{HARDWARE_PLATFORM}"
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
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_clm.py \
                                    --do_train \
                                    --model_type=gpt2 \
                                    --tokenizer_name=meta_data/v100_tokenizer \
                                    --output_dir=%s/meta_data/clm_gen_best_%s \
                                    --dataset_name=%s/meta_data/%s_gen_best \
                                    --per_device_train_batch_size=4 \
                                    \
                                    --overwrite_output_dir=True \
                                    --logging_steps=10 \
                                    --num_train_epochs=10 \
                                    --remove_unused_columns=False \
                                    --learning_rate=5e-5 \
                                    --model_name_or_path=meta_data/clm_gen_v100 \
                                    --lr_scheduler_type=constant
                                    
if [ $? -ne 0 ]; then
  curl https://diyi.site/ma\?text\=finish_run_train_clm_best_v100 --noproxy diyi.site
fi

                                    # 4层是64,128
                                    # --save_strategy=epoch \


date
} |& tee -a %s 
}' 
""" % (session_name,CURRENT_DATASET_FOLDER,HARDWARE_PLATFORM,CURRENT_DATASET_FOLDER,HARDWARE_PLATFORM,log_file)

# 使用 subprocess 运行命令
subprocess.Popen(cmd, shell=True)