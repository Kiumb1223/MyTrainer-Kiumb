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
import trackeval 
from multiprocessing import freeze_support
from torch.utils.tensorboard import SummaryWriter
#  tensorboard --logdir 'experiments\tb_logs'
if __name__ == '__main__':

    dataset_name = 'MOT17'
    cfg   = get_config()

    with open(cfg.JSON_PATH,'r') as f:
        data_json = json.load(f)

    tbwriter = SummaryWriter(
        os.path.join(cfg.WORK_DIR,'tb_logs')
    )

    freeze_support()

    eval_config = {
        'USE_PARALLEL':False,
        'NUM_PARALLEL_CORES':8,
        'PRINT_RESULTS': False,
        'PRINT_ONLY_COMBINED': True,
        'PRINT_CONFIG': False,
        'TIME_PROGRESS': False,
        'DISPLAY_LESS_PROGRESS': False,

        'OUTPUT_SUMMARY': False,
        'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
        'OUTPUT_DETAILED': False,
        'PLOT_CURVES': False,

    }
    evaluator = trackeval.Evaluator(eval_config)

    metrics_list = [trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()]


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
    
    
    record_metrics_list = {
        'HOTA':['HOTA','DetA','AssA'],
        'CLEAR':['MOTA','MOTP'],
        'Identity':['IDF1','IDR','IDP'],
    }
    for type, tracker in raw_results.items():
        for tracker_name , metrics_per_seq in tracker.items():
            print(f"Tracker:{tracker_name}")
            for cls ,number_per_metrics in metrics_per_seq['COMBINED_SEQ'].items():
                for metrics,index_list in record_metrics_list.items():
                    for index in index_list:
                        print(f"{index}(%):[{number_per_metrics[metrics][index].mean() * 100 :.2f}]")
                        tbwriter.add_scalar(f'Evalutions\{index}',number_per_metrics[metrics][index].mean())
        
    # freeze_support()
    # #---------------------------------#
    # # 配置评估器
    # #---------------------------------#
    # eval_config = {
    #     'USE_PARALLEL':False,
    #     'NUM_PARALLEL_CORES':8,
    # }
    # evaluator = trackeval.Evaluator(eval_config)

    # #---------------------------------#
    # #  配置指标
    # #---------------------------------#
    # metrics_list = [trackeval.metrics.HOTA(),trackeval.metrics.CLEAR(),trackeval.metrics.Identity()]
    
    # #---------------------------------#
    # #  配置数据格式以及文件路径等参数
    # #  按照MOT17格式进行存放数据
    # #  i.e. 
    # #    testVideo                               # GT_FOLDER
    # #       | ---  2024_0909_160937              # SEQ_INFO[key]
    # #                   | --  gt
    # #                          | --- gt.txt      # MOT数据存放格式  如果改变可以修改 GT_LOC_FORMAT
    # #
    # #    output                                  # TRACKERS_FOLDER  ## 可以用跟踪器的名字来定义名字，但是这里只是刚开始摸索，所以直接OUTPUT了
    # #       | ---  2024_0909_160937              # TRACKERS_TO_EVAL
    # #                 |(我这里没有建立文件夹)      # TRACKER_SUB_FOLDER
    # #                | ---- 2024_0909_160937.txt  # SEQ_INFO[key].txt
    # #---------------------------------#

    # dataset_config = {

    #     # 真值文件路径设置
    #     'GT_FOLDER':r'testVideo',  
    #     'SEQ_INFO':{             # 填写 自搭数据的文件名及总帧数
    #         '2024_0909_160937':1897, 
    #     },

    #     # 跟踪器输出结果文件路径设置
    #     'TRACKERS_FOLDER':r'testVideo\trackResult',
    #     'TRACKERS_TO_EVAL':[ 'gcnnmatch'
    #                          ],
    #     'TRACKER_SUB_FOLDER':r'',
    #     'SKIP_SPLIT_FOL': True,  # 自用数据所需设置为True
    # }
    # dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]


    # raw_results , messages = evaluator.evaluate(dataset_list,metrics_list)