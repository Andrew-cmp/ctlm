import numpy as np
import pandas as pd 
import argparse
import os
from meta_common import register_data_path, hold_out_task_files, get_jsondatabase_top1
import tvm.meta_schedule as ms
from meta_common import yield_hold_out_five_files
from glob import glob
import shutil
import logging
from tvm.target import Target
import pickle
import hashlib
import copy
import json
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--choice", required=True,type=str, choices = ["dump_programs","11"],help="Pick out the top k"
    )
    return parser.parse_args()
script_args = _parse_args()
def dump_programs():
    target_path = "dataset/measured/"
    target_dirs = glob(os.path.join(target_path,"sub*_*"))
    target_subgraph_dict = {}
    for target_dir in target_dirs:
        basename = os.path.basename(target_dir)
        target_name = basename.split("_")[-1]
        target_subgraph_tuning_record = {}
        target_measured_trace_dirs = glob(os.path.join(target_dir,'*'))
        for target_measured_trace_dir in target_measured_trace_dirs:
            database = ms.database.JSONDatabase(work_dir=target_measured_trace_dir) 
            subgraph_name = os.path.basename(target_measured_trace_dir)
            ###get_all_tuning_records是按照run sec排过序的屮
            # tuning_records = database.get_all_tuning_records()
            # tuning_records_list = []
            # for index,tuning_record in enumerate(tuning_records):
            #     trace_str = str(tuning_record.trace)
            #     run_secs = float(f"{float(tuning_record.run_secs[0]):.5g}")
            #     tuning_records_list.append([index,trace_str,run_secs])
            tuning_records_without_rank = database.get_all_tuning_records_without_rank()
            tuning_records_without_rank_list = []
            for index,tuning_record in enumerate(tuning_records_without_rank):
                trace_str = str(tuning_record.trace)
                run_secs = float(f"{float(tuning_record.run_secs[0]):.5g}")
                tuning_records_without_rank_list.append([index,trace_str,run_secs])
            target_subgraph_tuning_record.update({subgraph_name:tuning_records_without_rank_list})
        target_subgraph_dict.update({target_name:target_subgraph_tuning_record})
    ##上面开销过高，先存到json文件中
    f_save = open(os.path.join(target_path,'dict_file.json'),'w')
    json.dump(target_subgraph_dict,f_save,ensure_ascii=False)
    f_save.close()
    
def post_process_and_check():
    # pkl_path = "dataset/measured/dict_file.pkl"
    # f_read = open(pkl_path,'rb')
    # dict2 = pickle.load(f_read)
    
    pkl_path = "dataset/measured/dict_file.json"
    f_read = open(pkl_path,'r')
    dict2 = json.load(f_read)
    
    ##
    total_subgraphs_num = {}
    for target_name, subgraphs in dict2.items():
        for subgraphs_name, _ in subgraphs.items():
            if(total_subgraphs_num.get(subgraphs_name)):
                total_subgraphs_num[subgraphs_name] = total_subgraphs_num[subgraphs_name]+1
            else:
                total_subgraphs_num[subgraphs_name] = 1
    target_num =len(dict2)
    abandon_subgraph = []
    ##所有target的子图取交集
    for subgraph_name,num in total_subgraphs_num.items():
        if(num != target_num):
            print(f"{subgraph_name} will be abandoned")
            abandon_subgraph.append(subgraph_name)
    
    #得到标准的trace
    standard_subgraph = dict2["v100"]
    standard_subgraphs_trace_order = {}
    for subgraph_name, subgraph_trace_run_sec in standard_subgraph.items():
        if(subgraph_name in abandon_subgraph):
            continue
        subgraph_trace_order = []
        for index,trace_run_sec in enumerate(subgraph_trace_run_sec):
            subgraph_trace_order.extend([trace_run_sec])
        standard_subgraphs_trace_order.update({subgraph_name:subgraph_trace_order})
        
    
    dict2_ordered = {}
    ##检查各个target的各个子图的trace顺序和stand trace是否一致
    for target_name, subgraphs in dict2.items():
        target_subgraph_tuning_record_orderd = {}
        for subgraph_name, subgraph_trace_run_sec in subgraphs.items():
            if(subgraph_name in abandon_subgraph):
                continue
            stand_subgraphs_trace_order = standard_subgraphs_trace_order[subgraph_name]
            this_target_stand_subgraphs_trace_order = []
            for index,trace_run_sec in enumerate(subgraph_trace_run_sec):
                flag = 0
                for stand_subgraph_trace_order in stand_subgraphs_trace_order:
                    if(trace_run_sec[1] == stand_subgraph_trace_order[1]):
                        #print(f"{trace_run_sec[0]} and {stand_subgraph_trace_order[0]} yes")
                        this_target_stand_subgraphs_trace_order.append(trace_run_sec)
                        assert(flag == 0),f"{trace_run_sec[0]} find two"
                        flag = 1
                assert(flag == 1),f"{trace_run_sec[0]} no match"
                # 排序得到rank结果
            this_target_stand_subgraphs_trace_order.sort(key=lambda x: x[2])
            for rank, this_target_stand_subgraph_trace_order in enumerate(this_target_stand_subgraphs_trace_order):
                this_target_stand_subgraph_trace_order.append(rank)
                #然后再以index进行排序
            this_target_stand_subgraphs_trace_order.sort(key=lambda x: x[0])
            target_subgraph_tuning_record_orderd.update({subgraph_name:this_target_stand_subgraphs_trace_order})
        print(f"{target_name} has done")    
        dict2_ordered.update({target_name:target_subgraph_tuning_record_orderd})
        
    check_ordered(dict2_ordered)
    f_save = open("dataset/measured/dict_file_ordered.json",'w')
    json.dump(dict2_ordered,f_save,ensure_ascii=False)
    f_save.close()
def check_ordered(dict2=None):
    if(dict2 == None):
        json_path = "dataset/measured/dict_file_ordered.json"
        f_read = open(json_path,'r')
        dict2 = json.load(f_read)
    else:
        pass
    #得到标准的trace
    standard_subgraph = dict2["v100"]
    standard_subgraphs_trace_order = {}
    for subgraph_name, subgraph_trace_run_sec in standard_subgraph.items():
        subgraph_trace_order = []
        for index,trace_run_sec in enumerate(subgraph_trace_run_sec):
            subgraph_trace_order.extend([trace_run_sec])
        standard_subgraphs_trace_order.update({subgraph_name:subgraph_trace_order})
    
    for target_name, subgraphs in dict2.items():
        assert(len(subgraphs) == 997)
        for subgraph_name, subgraph_trace_run_sec in subgraphs.items():
            stand_subgraphs_trace_order = standard_subgraphs_trace_order[subgraph_name]
            assert(len(subgraph_trace_run_sec) == 50)
            for index,trace_run_sec in enumerate(subgraph_trace_run_sec):
                assert(trace_run_sec[1] == stand_subgraphs_trace_order[index][1]),f"{target_name}'s {subgraph_name} not match"
def transfer_to_df():
    json_path = "dataset/measured/dict_file_ordered.json"
    f_read = open(json_path,'r')
    dict2 = json.load(f_read)
    
                
def main():
    if(script_args.choice == "dump_programs"):
        dump_programs()
    elif(script_args.choice == "post_process_and_check"):
        post_process_and_check()
    elif(script_args.choice =="dataframe"):
        transfer_to_df()
if __name__ == "__main__":
    main()     