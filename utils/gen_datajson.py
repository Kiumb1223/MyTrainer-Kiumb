import os
import json 
from configs.config import get_config

def main():
    cfg = get_config()
    save_json = {
        'train_seq':{},
        'valid_seq':{}
    }

    for data_type in os.listdir(cfg.DATA_DIR):

        save_json['train_seq'][data_type] = {'seq_name':[],'start_frame':[],'end_frame':[]}
        save_json['valid_seq'][data_type] = {'seq_name':[],'start_frame':[],'end_frame':[]}

        if data_type in ['MOT17','MOT20']: # half-frame data for train  and half-frame data for validation 
            for seq in os.listdir(os.path.join(cfg.DATA_DIR,data_type,'train')):
                if not os.path.isdir(os.path.join(cfg.DATA_DIR,data_type,'train',seq)):
                    continue
                seq_path = os.path.join(cfg.DATA_DIR,data_type,'train',seq)
                ini_path = os.path.join(seq_path,'seqinfo.ini')
                with open(ini_path,'r') as f:
                    lines_split = [ l.split('=') for l in f.read().splitlines()[1:]]
                    info_dict  = dict(s for s in lines_split if isinstance(s,list) and len(s) == 2)
                
                save_json['train_seq'][data_type]['seq_name'].append(seq)
                save_json['train_seq'][data_type]['start_frame'].append(1)
                save_json['train_seq'][data_type]['end_frame'].append(int(info_dict['seqLength'])//2)

                save_json['valid_seq'][data_type]['seq_name'].append(seq)
                save_json['valid_seq'][data_type]['start_frame'].append(int(info_dict['seqLength'])//2 + 1 )
                save_json['valid_seq'][data_type]['end_frame'].append(int(info_dict['seqLength']))

        # elif data_type in ['DanceTrack']:
        #     for train_seq in os.listdir(os.path.join(cfg.DATA_DIR,data_type,'train')):
        #         if not os.path.isdir(os.path.join(cfg.DATA_DIR,data_type,'train',train_seq)):
        #             continue
        #         seq_path = os.path.join(cfg.DATA_DIR,data_type,'train',train_seq)
        #         ini_path = os.path.join(seq_path,'seqinfo.ini')
        #         with open(ini_path,'r') as f:
        #             lines_split = [ l.split('=') for l in f.read().splitlines()[1:]]
        #             info_dict  = dict(s for s in lines_split if isinstance(s,list) and len(s) == 2)

        #         save_json['train_seq'][data_type]['seq_name'].append(train_seq)
        #         save_json['train_seq'][data_type]['start_frame'].append(1)
        #         save_json['train_seq'][data_type]['end_frame'].append(int(info_dict['seqLength']))          

        #     for valid_seq in os.listdir(os.path.join(cfg.DATA_DIR,data_type,'val')):
        #         if not os.path.isdir(os.path.join(cfg.DATA_DIR,data_type,'val',valid_seq)):
        #             continue
        #         seq_path = os.path.join(cfg.DATA_DIR,data_type,'val',valid_seq)
        #         ini_path = os.path.join(seq_path,'seqinfo.ini')
        #         with open(ini_path,'r') as f:
        #             lines_split = [ l.split('=') for l in f.read().splitlines()[1:]]
        #             info_dict  = dict(s for s in lines_split if isinstance(s,list) and len(s) == 2)

        #         save_json['valid_seq'][data_type]['seq_name'].append(valid_seq)
        #         save_json['valid_seq'][data_type]['start_frame'].append(1)
        #         save_json['valid_seq'][data_type]['end_frame'].append(int(info_dict['seqLength']))
        # else:
        #     raise ValueError('data_type not supported')
        
        with open(cfg.JSON_PATH,'w') as json_file:
            json.dump(save_json, json_file,indent=4)
        print(f'json file saved to {cfg.JSON_PATH}')
if __name__ == '__main__':
    main()
