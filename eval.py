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
import numpy as np
import time
import datetime
from tqdm import tqdm
from loguru import logger
from configs.config import get_config
from models.graphModel import GraphModel
from utils.visualize import plot_tracking
from models.graphTracker import TrackManager
from torchvision.io.image import read_image

@logger.catch
def main():
    cfg   = get_config()
    model = GraphModel(cfg)
    trackManager = TrackManager(model,cfg.DEVICE,cfg.PATH_TO_WEIGHTS,
                    cfg.RESIZE_TO_CNN,cfg.MATCH_THRESH,cfg.Det2Tra_CONF,
                    cfg.CNT_TO_ACTIVE,cfg.MAX_CNT_TO_DEAD,cfg.FEATURE_LIST_SIZE)
    
    #---------------------------------#
    #  prepare data 
    #  and only support MOT-format input data , i.e. /img1 and /det -> [<frame_id>, <id>, <x>, <y>, <w>, <h>, <score>]
    #  and output the video annotated with the tracking result and text file for evaluation
    #---------------------------------#
    test_root_dir  = r'testVideo'
    output_dir     = os.path.join(test_root_dir,'trackResult')
    seq_name = ['2024_0909_160937']
                # 'MOT17-05-FRCNN','MOT17-09-FRCNN','MOT17-10-FRCNN',
                # 'MOT17-11-FRCNN','MOT17-13-FRCNN',]
    # seq_name = ['2024_0909_160937','MOT17-02-FRCNN','MOT17-04-FRCNN',
    #             'MOT17-05-FRCNN','MOT17-09-FRCNN','MOT17-10-FRCNN',
    #             'MOT17-11-FRCNN','MOT17-13-FRCNN',]
    for seq in seq_name:
        seq_det_path  = os.path.join(test_root_dir,seq,'det','2024_0909_160937(yolov8-det).txt')
        seq_img_dir   = os.path.join(test_root_dir,seq,'img1')
        output_txt    = os.path.join(output_dir,seq)
        output_video  = os.path.join(output_dir,seq,'video')
        os.makedirs(output_txt,exist_ok=True)
        os.makedirs(output_video,exist_ok=True)

        seq_info_path = os.path.join(test_root_dir,seq,'seqinfo.ini')
        with open(seq_info_path,'r') as f:
            lines_split = [ l.split('=') for l in f.read().splitlines()[1:]]
            info_dict  = dict(s for s in lines_split if isinstance(s,list) and len(s) == 2)

        vid_writer = cv2.VideoWriter(output_video + os.sep + f'{seq}.avi', cv2.VideoWriter_fourcc(*'XVID'),
                        int(info_dict['frameRate']), (int(info_dict['imWidth']),int(info_dict['imHeight'])))
        txt_res = []
        detections = np.loadtxt(seq_det_path,delimiter=',')
        total_frame = np.max(detections[:,0]).astype(int)
        # BUG WHEN no dets 
        elapsed_times = []
        for frame_id in tqdm(range(1,total_frame+1),total=total_frame,desc=f'processing seq - {seq} ',unit='frame'):
            frame_det = detections[(detections[:,0] == frame_id) 
                                   &(detections[:,2] >= 0 )&(detections[:,3] >= 0 )
                                   &(detections[:,6] > cfg.MIN_DET_CONF)]
            if frame_det.size == 0 :
                logger.info(f"no dets in {frame_id}-th frame")
                continue
            start = time.perf_counter()
            img_data  = read_image(os.path.join(seq_img_dir,f'{frame_id:06d}.jpg'))
            img_cv    = cv2.imread(os.path.join(seq_img_dir,f'{frame_id:06d}.jpg'))
            trackers_list = trackManager.graph_matching(frame_id,frame_det[:,2:],img_data) # need to be careful with the input format
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
        logger.info(f'{seq} is done and saved to {output_txt} and {output_video}')
        logger.info(f'{seq}  - Average elapsed time: {datetime.timedelta(seconds=sum(elapsed_times) / len(elapsed_times))}')
if __name__ == '__main__':
    main()