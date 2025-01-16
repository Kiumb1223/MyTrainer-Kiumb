#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     eval.py
@Time     :     2024/12/03 16:48:22
@Author   :     Louis Swift
@Desc     :     
'''
import os 
import cv2
import sys
import time
import datetime
import numpy as np
from tqdm import tqdm
from loguru import logger
import torchvision.io.image as I
from configs.config import get_config
from models.graphModel import GraphModel
from utils.visualize import plot_tracking
from multiprocessing import freeze_support
from models.graphTracker import TrackManager

sys.path.append(os.path.join('thirdparty','TrackEval'))
import trackeval 
@logger.catch
def main():
    
    # input you wanna test
    # dataset_name   = 'self-dataset'
    test_root_dir  = 'testVideo'
    seq_name_list  = ['2024_0909_160937']
    cfg   = get_config()
    model = GraphModel(cfg.MODEL_YAML_PATH)
    trackManager = TrackManager(model,cfg.DEVICE,cfg.PATH_TO_WEIGHTS,
        cfg.FUSION_METHOD,cfg.EMA_LAMBDA,
        cfg.RESIZE_TO_CNN,cfg.MATCH_THRESH,cfg.Det2Tra_CONF,
        cfg.CNT_TO_ACTIVE,cfg.CNT_TO_SLEEP,cfg.MAX_CNT_TO_DEAD,cfg.FEATURE_LIST_SIZE)
    
    #---------------------------------#
    #  prepare data 
    #  and only support MOT-format input data , i.e. /img1 and /det -> [<frame_id>, <id>, <x>, <y>, <w>, <h>, <score>]
    #  and output the video annotated with the tracking result and text file for evaluation
    #---------------------------------#

    output_dir     = 'trackResult'
    tracker_name   = 'myTracker'
    # move_to_path   = os.path.join(data_json['Trackeval']['TRACKERS_FOLDER'],tracker_name)

    for cnt, seq in enumerate(seq_name_list):
        seq_det_path  = os.path.join( test_root_dir,seq,'det','2024_0909_160937(yolov8-det).txt')
        seq_img_dir   = os.path.join( test_root_dir,seq,'img1')
        seq_info_path = os.path.join( test_root_dir,seq,'seqinfo.ini')   
        output_txt    = os.path.join(output_dir,tracker_name,seq)
        output_video  = os.path.join(output_dir,tracker_name,seq,'video')
        os.makedirs(output_txt,exist_ok=True)
        os.makedirs(output_video,exist_ok=True)
        with open(seq_info_path,'r') as f:
            lines_split = [ l.split('=') for l in f.read().splitlines()[1:]]
            info_dict  = dict(s for s in lines_split if isinstance(s,list) and len(s) == 2)

        vid_writer = cv2.VideoWriter(output_video + os.sep + f'{seq}.avi', cv2.VideoWriter_fourcc(*'XVID'),
                        int(info_dict['frameRate']), (int(info_dict['imWidth']),int(info_dict['imHeight'])))
        logger.info(f'Seq {seq} info : [Height - {info_dict["imHeight"]};Width - {info_dict["imWidth"]};FrameRate - {info_dict["frameRate"]}] ')
        txt_res = []
        detections = np.loadtxt(seq_det_path,delimiter=',')
        min_frame = np.min(detections[:,0]).astype(int)
        max_frame = np.max(detections[:,0]).astype(int)
        # BUG WHEN no dets 
        elapsed_times = []
        for frame_id in tqdm(range(min_frame,max_frame+1),total=max_frame,desc=f'processing seq - {seq} ',unit='frame'):
            frame_det = detections[(detections[:,0] == frame_id) & 
                                   (detections[:,6] > cfg.MIN_DET_CONF)]
            if frame_det.size == 0 :
                logger.info(f"no dets in {frame_id}-th frame")
                continue
            start = time.perf_counter()
            img_data  = I.read_image(os.path.join(seq_img_dir,f'{frame_id:06d}.jpg'))
            img_cv    = img_data.clone().permute(1,2,0).numpy()[...,::-1].astype(np.uint8)
            
            trackers_list = trackManager.update(frame_id,frame_det[:,2:],img_data) # need to be careful with the input format
            
            end = time.perf_counter()
            elapsed_times.append(end-start)
            # print(datetime.timedelta(seconds=end-start))
            online_tlwhs,online_ids,online_confs = [],[],[]
            # print(f'number of tracked - {len(trackers_list)}')
            for tracker in trackers_list:
                tlwh = tracker.tlwh
                tid  = tracker.track_id
                conf = tracker.conf
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_confs.append(conf)
                txt_res.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{conf:.2f},-1,-1,-1\n")

            online_im = plot_tracking(
                img_cv,online_tlwhs, online_ids, frame_id=frame_id, fps=int(info_dict['frameRate'])
            )
            vid_writer.write(online_im)
        with open(os.path.join(output_txt,f'{seq}.txt'),'w') as f:
            f.writelines(txt_res)


        trackManager.clean_cache()
        vid_writer.release()

        average_elapsed_time = sum(elapsed_times) / len(elapsed_times)
        frame_rate = 1 / average_elapsed_time
        logger.info(f'{seq} is done and saved to {output_txt} and {output_video}')
        logger.info(f'{seq}  - Average elapsed time: {datetime.timedelta(seconds=average_elapsed_time)}')
        logger.info((f'{seq} - Frame rate: {frame_rate:.2f} fps'))
        

    #---------------------------------#
    #  NOW test
    #---------------------------------#
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
    
    # ---------------------------------#
    # 配置数据格式以及文件路径等参数
    # 按照MOT17格式进行存放数据
    # i.e. 
    # testVideo                               # GT_FOLDER
    #     | ---  2024_0909_160937              # SEQ_INFO[key]
    #                 | --  gt
    #                         | --- gt.txt      # MOT数据存放格式  如果改变可以修改 GT_LOC_FORMAT
    
    # output                                  # TRACKERS_FOLDER  ## 可以用跟踪器的名字来定义名字，但是这里只是刚开始摸索，所以直接OUTPUT了
    #     | ---  2024_0909_160937              # TRACKERS_TO_EVAL
    #                 |(我这里没有建立文件夹)      # TRACKER_SUB_FOLDER
    #             | ---- 2024_0909_160937.txt  # SEQ_INFO[key].txt
    # ---------------------------------#

    dataset_config = {

        # 真值文件路径设置
        'GT_FOLDER':r'testVideo',  
        'SEQ_INFO':{             # 填写 自搭数据的文件名及总帧数
            '2024_0909_160937':1897, 
        },

        # 跟踪器输出结果文件路径设置
        'TRACKERS_FOLDER':rf'{output_dir}',
        'TRACKERS_TO_EVAL':[ 
            f'{tracker_name}'
            ],
        'TRACKER_SUB_FOLDER':r'2024_0909_160937',
        'SKIP_SPLIT_FOL': True,  # 自用数据所需设置为True
    }
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]


    raw_results , messages = evaluator.evaluate(dataset_list,metrics_list)
    record_metrics_list = {
    'HOTA':['HOTA','DetA','AssA'],
    'Identity':['IDF1','IDR','IDP'],
    'CLEAR':['MOTA','MOTP'],
    }
    for type, tracker in raw_results.items():
        for tracker_name , metrics_per_seq in tracker.items():
            print(f"Tracker:{tracker_name}")
            for cls ,number_per_metrics in metrics_per_seq['COMBINED_SEQ'].items():
                for metrics,index_list in record_metrics_list.items():
                    for index in index_list:
                        print(f"{index}(%):[{number_per_metrics[metrics][index].mean() * 100 :.2f}]")
if __name__ == '__main__':
    main()