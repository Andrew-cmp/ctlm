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
from scipy.stats import kendalltau
from scipy.stats import weightedtau
from count_inversions import count_inversions
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--choice", required=True,type=str, choices = ["dump_programs","dataframe","post_process_and_check","inversions","weighted_inversions"],help="Pick out the top k"
    )
    return parser.parse_args()
script_args = _parse_args()
# #将所有的database转存到json文件中，以备后续读取。
# #按照每个硬件下排序
# def dump_programs():    
#     target_path = "dataset/measured/"
#     target_dirs = glob(os.path.join(target_path,"sub*_*"))
#     target_dict = {}
#     for target_dir in target_dirs:
#         basename = os.path.basename(target_dir)
#         target_name = basename.split("_")[-1]
#         target_measured_trace_dirs = glob(os.path.join(target_dir,'*'))
#         target_subgraph_list = []
#         for index_out,target_measured_trace_dir in enumerate(target_measured_trace_dirs):
#             database = ms.database.JSONDatabase(work_dir=target_measured_trace_dir) 
#             subgraph_name = os.path.basename(target_measured_trace_dir)
#             tuning_records_without_rank = database.get_all_tuning_records_without_rank()
#             tuning_records_without_rank_list = []
#             for index_inner,tuning_record in enumerate(tuning_records_without_rank):
#                 trace_str = str(tuning_record.trace)
#                 run_secs = float(f"{float(tuning_record.run_secs[0]):.5g}")
#                 tuning_records_without_rank_list.append([index_out*50+index_inner,trace_str,run_secs])
#             target_subgraph_list.extend(tuning_records_without_rank_list)
#         target_dict.update({target_name:target_subgraph_list})
#     ##上面开销过高，先存到json文件中
#     f_save = open(os.path.join(target_path,'dict_file_per_target.json'),'w')
#     json.dump(target_dict,f_save,ensure_ascii=False)
#     f_save.close()

#将所有的database转存到json文件中，以备后续读取。
#按照每个硬件下的每个组图进行排序
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
    f_save = open(os.path.join(target_path,'dict_file .json'),'w')
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
    
    dict2_per_target_ordered = {}
    
    for target_name, subgraphs in dict2_ordered.items():
        target_tuning_records_orderd = []
        for index_outer,(subgraph_name, subgraph_trace_run_sec) in enumerate(subgraphs.items()):
            for index_inner,trace_run_sec in enumerate(subgraph_trace_run_sec):
                target_tuning_records_orderd.append([index_outer*50+index_inner,*trace_run_sec,subgraph_name])
        target_tuning_records_orderd.sort(key=lambda x: x[3])
        for rank, target_tuning_record_orderd in enumerate(target_tuning_records_orderd):
            target_tuning_record_orderd.append(rank)
            #然后再以index进行排序
        target_tuning_records_orderd.sort(key=lambda x: x[0])
        print(f"{target_name} has done")    
        dict2_per_target_ordered.update({target_name:target_tuning_records_orderd})
    f_save = open("dataset/measured/dict_file_per_target_ordered.json",'w')
    json.dump(dict2_per_target_ordered,f_save,ensure_ascii=False)
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
    json_path = "dataset/measured/dict_file_per_target_ordered.json"
    f_read = open(json_path,'r')
    dict2 = json.load(f_read)
    
    df_list = []
    for target_name, subgraphs_trace_run_sec in dict2.items():
        col = ["index_outer","index_inner","sec_data","rank_inner_data","subgraph_name","rank_outer_data"]
        col_data_dict = {}
        index_outer = []
        index_inner = []
        sec_data = []
        rank_inner_data = []
        subgraph_name = []
        rank_outer_data = []
        for subgraph_trace_run_sec in subgraphs_trace_run_sec:
            index_outer.append(subgraph_trace_run_sec[0])
            index_inner.append(subgraph_trace_run_sec[1])
            sec_data.append(subgraph_trace_run_sec[3])
            rank_inner_data.append(subgraph_trace_run_sec[4])
            subgraph_name.append(subgraph_trace_run_sec[5])
            rank_outer_data.append(subgraph_trace_run_sec[6])
        col_data_dict.update({f"index_outer":index_outer})
        col_data_dict.update({f"index_inner":index_inner})
        col_data_dict.update({f"sec_data":sec_data})
        col_data_dict.update({f"rank_inner_data":rank_inner_data})
        col_data_dict.update({f"subgraph_name":subgraph_name})
        col_data_dict.update({f"rank_outer_data":rank_outer_data})
        df = pd.DataFrame(col_data_dict,columns = col)
        print(df)
        df_list.append(df)
        df.to_csv(f"dataset/measured/statistics_per_target_each_target/{target_name}.csv")


# 使用kendalltau函数直接计算不同target之间的距离，而不是计算每个target每个subgraph的逆序数。
def inversions():
    json_path = "dataset/measured/dict_file_per_target_ordered.json"
    f_read = open(json_path,'r')
    dict2 = json.load(f_read)
    
    dict_target_rank = {}
    target_list = []
    for target_name, subgraphs_trace_run_sec in dict2.items():
        target_list.append(target_name)
        rank = []
        for subgraph_trace_run_sec in subgraphs_trace_run_sec:
            rank.append(subgraph_trace_run_sec[6])
        
        dict_target_rank.update({f"{target_name}":rank})
    distance = np.zeros((len(target_list), len(target_list)))
    for i,target_name_outer in enumerate(target_list):
        for j,target_name_inner in enumerate(target_list):
            rank_outer = dict_target_rank[target_name_outer]
            rank_inner = dict_target_rank[target_name_inner]
            tau, p_value = kendalltau(rank_outer, rank_inner)
            distance[i][j] = tau
    df = pd.DataFrame(data=distance,index = target_list,columns=target_list)
    print(df)
    df.to_csv(f"dataset/measured/statistics_per_target_each_target/distance.csv")
def weighted_inversions():
    json_path = "dataset/measured/dict_file_per_target_ordered.json"
    f_read = open(json_path,'r')
    dict2 = json.load(f_read)
    
    dict_target_rank = {}
    target_list = []
    for target_name, subgraphs_trace_run_sec in dict2.items():
        target_list.append(target_name)
        rank = []
        for subgraph_trace_run_sec in subgraphs_trace_run_sec:
            rank.append(subgraph_trace_run_sec[6])
        
        dict_target_rank.update({f"{target_name}":rank})
    distance = np.zeros((len(target_list), len(target_list)))
    for i,target_name_outer in enumerate(target_list):
        for j,target_name_inner in enumerate(target_list):
            rank_outer = dict_target_rank[target_name_outer]
            rank_inner = dict_target_rank[target_name_inner]
            tau, p_value = weightedtau(rank_outer, rank_inner，rank=False)
            distance[i][j] = tau
    df = pd.DataFrame(data=distance,index = target_list,columns=target_list)
    print(df)
    df.to_csv(f"dataset/measured/statistics_per_target_each_target/distance.csv")
                 
def main():
    if(script_args.choice == "dump_programs"):
        dump_programs()
    elif(script_args.choice == "post_process_and_check"):
        post_process_and_check()
    elif(script_args.choice =="dataframe"):
        transfer_to_df()
    elif(script_args.choice =="inversions"):
        inversions()
    elif(script_args.choice =="weighted_inversions"):
        weighted_inversions()
if __name__ == "__main__":
    main()     