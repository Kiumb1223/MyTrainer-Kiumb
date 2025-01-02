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
import time
import json 
import shutil
import datetime
import numpy as np
from tqdm import tqdm
from loguru import logger
import torchvision.io.image as I
from configs.config import get_config
from models.graphModel import GraphModel
from utils.visualize import plot_tracking
from models.graphTracker import TrackManager

@logger.catch
def main():
    
    # input you wanna test
    dataset_name   = 'MOT17'


    cfg   = get_config()
    model = GraphModel(cfg.MODEL_YAML_PATH)
    trackManager = TrackManager(model,cfg.DEVICE,cfg.PATH_TO_WEIGHTS,
                    cfg.RESIZE_TO_CNN,cfg.MATCH_THRESH,cfg.Det2Tra_CONF,
                    cfg.CNT_TO_ACTIVE,cfg.CNT_TO_SLEEP,cfg.MAX_CNT_TO_DEAD,cfg.FEATURE_LIST_SIZE)
    
    with open(cfg.JSON_PATH,'r') as f:
        data_json = json.load(f)
    #---------------------------------#
    #  prepare data 
    #  and only support MOT-format input data , i.e. /img1 and /det -> [<frame_id>, <id>, <x>, <y>, <w>, <h>, <score>]
    #  and output the video annotated with the tracking result and text file for evaluation
    #---------------------------------#

    output_dir     = 'trackResult'
    tracker_name   = 'myTracker'
    # move_to_path   = os.path.join(data_json['Trackeval']['TRACKERS_FOLDER'],tracker_name)
    os.makedirs(output_dir,exist_ok=True)


    seq_name_list    = data_json['valid_seq'][dataset_name]['seq_name']
    for cnt, seq in enumerate(seq_name_list):
        # seq_det_path  = os.path.join(test_root_dir,seq,'det','2024_0909_160937(yolov8-det).txt')
        if dataset_name in ['MOT17','MOT20']:
            test_root_dir = data_json['Trackeval']['GT_FOLDER']+os.sep+f"{dataset_name}-{data_json['valid_seq'][dataset_name]['Trackeval']['SPLIT_TO_EVAL']}"
        seq_det_path  = os.path.join( test_root_dir,seq,'det','det.txt')
        seq_img_dir   = os.path.join( test_root_dir,seq,'img1')
        seq_info_path = os.path.join( test_root_dir,seq,'seqinfo.ini')   
        output_txt    = os.path.join(output_dir,tracker_name,dataset_name,seq)
        output_video  = os.path.join(output_dir,tracker_name,dataset_name,seq,'video')
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

        move_to_folder = os.path.join(data_json['Trackeval']['TRACKERS_FOLDER'],f"{dataset_name}-{data_json['valid_seq'][dataset_name]['Trackeval']['SPLIT_TO_EVAL']}",tracker_name,'data')
        if cnt == 0 :
            shutil.rmtree(move_to_folder,ignore_errors=True)
        os.makedirs(move_to_folder,exist_ok=True)
        shutil.copy(output_txt+os.sep+f'{seq}.txt',move_to_folder)
        trackManager.clean_cache()
        vid_writer.release()

        average_elapsed_time = sum(elapsed_times) / len(elapsed_times)
        frame_rate = 1 / average_elapsed_time
        logger.info(f'{seq} is done and saved to {output_txt} and {output_video}')
        logger.info(f'{seq}  - Average elapsed time: {datetime.timedelta(seconds=average_elapsed_time)}')
        logger.info((f'{seq} - Frame rate: {frame_rate:.2f} fps'))
        
if __name__ == '__main__':
    main()