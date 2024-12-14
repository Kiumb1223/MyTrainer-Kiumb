#---------------------------------#
# class_name_to_class_id = {'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4, 'motorbike': 5,
#                           'non_mot_vehicle': 6, 'static_person': 7, 'distractor': 8, 'occluder': 9,
#                           'occluder_on_ground': 10, 'occluder_full': 11, 'reflection': 12, 'crowd': 13}
#---------------------------------#clear
import os
import sys
import json
from configs.config import get_config 
sys.path.append(os.path.join('thirdparty','Trackeval'))
import trackeval # env - py3.8

from multiprocessing import freeze_support


# def main():
#     dataset_name = 'MOT17'
#     cfg   = get_config()
#     with open(cfg.JSON_PATH,'r') as f:
#         data_json = json.load(f)

#     # os.system(f"python3 thirdparty/TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL {data_json['valid_seq'][dataset_name]['Trackeval']['SPLIT_TO_EVAL']}  "
#     os.system(f"python3 thirdparty\\TrackEval\\scripts\\run_mot_challenge.py --SPLIT_TO_EVAL {data_json['valid_seq'][dataset_name]['Trackeval']['SPLIT_TO_EVAL']}  "
#             f"--METRICS HOTA CLEAR Identity  --GT_FOLDER {data_json['Trackeval']['GT_FOLDER']} "
#             f"--SKIP_SPLIT_FOL True --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True "
#             f"--NUM_PARALLEL_CORES 8 --PLOT_CURVES False --BENCHMARK {dataset_name}"
#             f"--TRACKERS_FOLDER {data_json['Trackeval']['TRACKERS_FOLDER']}")
    
# if __name__ == '__main__':
#     main()
if __name__ == '__main__':
    dataset_name = 'MOT17'
    cfg   = get_config()
    with open(cfg.JSON_PATH,'r') as f:
        data_json = json.load(f)
    freeze_support()
    #---------------------------------#
    # 配置评估器
    #---------------------------------#
    eval_config = {
        'USE_PARALLEL':False,
        'NUM_PARALLEL_CORES':8,
    }
    evaluator = trackeval.Evaluator(eval_config)

    #---------------------------------#
    #  配置指标
    #---------------------------------#
    metrics_list = [trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()]
    
    #---------------------------------#
    #  配置数据格式以及文件路径等参数
    #  按照MOT17格式进行存放数据
    #  i.e. 
    #    testVideo                               # GT_FOLDER
    #       | ---  2024_0909_160937              # SEQ_INFO[key]
    #                   | --  gt
    #                          | --- gt.txt      # MOT数据存放格式  如果改变可以修改 GT_LOC_FORMAT
    #
    #    output                                  # TRACKERS_FOLDER  ## 可以用跟踪器的名字来定义名字，但是这里只是刚开始摸索，所以直接OUTPUT了
    #       | ---  2024_0909_160937              # TRACKERS_TO_EVAL
    #                 |(我这里没有建立文件夹)      # TRACKER_SUB_FOLDER
    #                | ---- 2024_0909_160937.txt  # SEQ_INFO[key].txt
    #---------------------------------#

    dataset_config = {

        # 真值文件路径设置
        'GT_FOLDER':data_json['Trackeval']['GT_FOLDER'],
        # 跟踪器输出结果文件路径设置
        'TRACKERS_FOLDER':data_json['Trackeval']['TRACKERS_FOLDER'],
        'SKIP_SPLIT_FOL': False,  # 自用数据所需设置为True
        'BENCHMARK':dataset_name,
        'SPLIT_TO_EVAL':data_json['valid_seq'][dataset_name]['Trackeval']['SPLIT_TO_EVAL'],
    }
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]


    raw_results , messages = evaluator.evaluate(dataset_list,metrics_list)
    # print(raw_results['HOTA'])
