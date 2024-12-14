
import argparse
import os, json, glob
import shutil


parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
parser.add_argument(
    "--task_dir",
    type=str,
    required=True,
)
parser.add_argument(
    "--move2dir",
    type=str,
    required=True,
)        
args = parser.parse_args()  # pylint: disable=invalid-name
task_dir = args.task_dir

dirs = glob.glob(f'{task_dir}/*winograd*')
for dir in dirs:
    src = os.path.join(task_dir,dir)
    dst = args.move2dir
    shutil.move(src,dst)
#python delete_wingord.py --task_dir=/home/hwhu/ctlm/experiment/dataset/to_measure_programs/a6000 --move2dir=/home/hwhu/ctlm/experiment/dataset/to_measure_programs_bak/remove_wingord