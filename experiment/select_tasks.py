import random
import glob

import shutil
import os
source_path = "dataset/to_measure_programs/a6000"
target_path = "dataset/to_measure_programs_t/tasks_be_removed"
# 设置随机种子
random.seed(66)

# 示例列表
data = glob.glob(os.path.join(source_path,"*"))

# 计算选中元素的数量，60% 的数量
num_elements_to_select = int(len(data) * 0.5)

# 随机选取 60% 的元素
selected_items = random.sample(data, num_elements_to_select)
print("选中的元素:", selected_items)
for dir in selected_items:
    task_name = os.path.basename(dir)
    shutil.move(dir,os.path.join(target_path,task_name))
