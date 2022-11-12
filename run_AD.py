import os
import sys
sys.path.append('../')
import pickle as pkl
import numpy as np
import glob
import yaml
from sklearn import metrics
from lib.anomaly_measures import *
from lib.utils.eval_utils import Evaluator
import matplotlib.pyplot as plt
from lib.utils.visualize_utils import vis_multi_prediction
import cv2
from vis import run

import PIL
from PIL import Image
import pandas as pd
import scipy.misc

import copy
from config.config import parse_args, visualize_config
import pdb

    
def main(args):
    # load data
    all_files = sorted(glob.glob(os.path.join(args.save_dir, '*.pkl')))
    print("Number of videos: ", len(all_files))
    
    # initialize evaluator and labels
    #evaluator = Evaluator(args, label_file=args.label_file)
    evaluator = Evaluator(args, label_file=False)

    #Load the fol-ego prediction results and compute AD measures
    all_mean_iou_anomaly_scores = {}
    all_min_iou_anomaly_scores = {}
    all_ego_anomaly_scores = {}
    all_mask_anomaly_scores = {}
    all_pred_std_mean_anomaly_scores = {}
    all_pred_std_max_anomaly_scores = {}

    anomalous = {}
    ano = {}
    mask_ano = {}
    
    for file_idx, fol_ego_file in enumerate(all_files):
        video_name = fol_ego_file.split('\\')[-1].split('.')[0]
        
        # csv 파일 생성.
        if os.path.isfile('C:/Users/rosha/Desktop/tad-IROS2019-master/csv/'+str(video_name)+'.csv'):
            os.remove('C:/Users/rosha/Desktop/tad-IROS2019-master/csv/'+str(video_name)+'.csv')
            print('csv file delete')
        
        header_data={'a':['frame_id'],
                     #'b':['anomaly_id'],
                     'c':['cx'],
                     'd':['cy'],
                     'e':['w'],
                     'f':['h']}
        df=pd.DataFrame(header_data)
        df.to_csv('C:/Users/rosha/Desktop/tad-IROS2019-master/csv/'+str(video_name)+'.csv', mode='a', index=False, header=False)
        
        
        image_folder = os.path.join(args.img_dir, video_name,'images')
        track_file = os.path.join(args.track_dir, video_name + '.npy')
        seq_folder = os.path.join(args.img_dir, video_name)
        det_file = os.path.join(args.img_dir, video_name, 'bbox.npy')
        video_len = len(glob.glob(os.path.join(image_folder, '*')))
        #video_len = evaluator.video_lengths[video_name]
        print(video_len)
        '''save anomaly scores in dictionary'''
        all_mean_iou_anomaly_scores[video_name] = np.zeros(video_len)
        all_min_iou_anomaly_scores[video_name] = np.zeros(video_len)
        all_ego_anomaly_scores[video_name] = np.zeros(video_len)
        all_mask_anomaly_scores[video_name] = np.zeros(video_len)
        all_pred_std_mean_anomaly_scores[video_name] = np.zeros(video_len)
        all_pred_std_max_anomaly_scores[video_name] = np.zeros(video_len)
       
        
        fol_ego_data = pkl.load(open(fol_ego_file,'rb'))
        
        count = 1
        
        for frame in fol_ego_data:
            '''compute iou metrics'''
            
            L_bbox = iou_metrics(frame['bbox_pred'],  # 1 - float(L_bbox)
                                frame['bbox_gt'],
                                multi_box='average', 
                                iou_type='average')
            all_mean_iou_anomaly_scores[video_name][frame['frame_id']] = L_bbox
            
            L_Mask, pred_mask, observed_mask = mask_iou_metrics(frame['bbox_pred'], 
                                                                frame['bbox_gt'], 
                                                                args.W, args.H, 
                                                                multi_box='latest')
            all_mask_anomaly_scores[video_name][frame['frame_id']+1] = L_Mask
            
            L_bbox = iou_metrics(frame['bbox_pred'], 
                                frame['bbox_gt'],
                                multi_box='average', 
                                iou_type='min')
            all_min_iou_anomaly_scores[video_name][frame['frame_id']] = L_bbox    
            
            
            L_pred_bbox_max, L_pred_bbox_mean, anomalous_object, _ = prediction_std(frame['bbox_pred'])
            all_pred_std_mean_anomaly_scores[video_name][frame['frame_id']] = L_pred_bbox_mean
            all_pred_std_max_anomaly_scores[video_name][frame['frame_id']] = L_pred_bbox_max
            
            
            #print(all_pred_std_max_anomaly_scores[video_name][frame['frame_id']])
            ##print(frame['frame_id'])
            #print('anomalus object id:', anomalous_object)
            #print("-----------")
            
            ano = iou_ano(frame['bbox_pred'], frame['bbox_gt'])
            mask_ano = mask_iou_ano(frame['bbox_pred'], 
                                frame['bbox_gt'], 
                                args.W, args.H, 
                                multi_box='latest')
            
            ##################################################################################
            ##################################################################################
            #1. ano -> iou로 anomalous_object 감지
            #2. anomalous_object -> std로 anomalous_object 감지
            #3. mask_ano -> mask_iou로 anomalous_object 감지
            
            #anomalous[frame['frame_id']] = ano 
            #anomalous[frame['frame_id']] = anomalous_object
            #anomalous[frame['frame_id']] = mask_ano
            ##################################################################################
            ##################################################################################
           
           
            ##################################################################################
            ##################################################################################
            
            # 일정 스코어 이상 이상치만 탐지하는 코드...
            
            #if  all_mean_iou_anomaly_scores[video_name][frame['frame_id']] >= 0.4:
            if  all_mask_anomaly_scores[video_name][frame['frame_id']] >= 0.2:
            #if  all_min_iou_anomaly_scores[video_name][frame['frame_id']] >= 0.7:
            #if  all_pred_std_mean_anomaly_scores[video_name][frame['frame_id']] >= 0.01:
            #if  all_pred_std_max_anomaly_scores[video_name][frame['frame_id']] >= 0.013:
                
            #    anomalous[frame['frame_id']] = ano 
            #    anomalous[frame['frame_id']] = anomalous_object
                anomalous[frame['frame_id']] = mask_ano
                
                # csv 파일 쓰기.
                
                try:                    
                    ori_data={'a':[frame['frame_id']],
                          #'b':[mask_ano],
                            'c':[float(frame['bbox_gt'][anomalous[frame['frame_id']]][0][0])],
                            'd':[float(frame['bbox_gt'][anomalous[frame['frame_id']]][0][1])],
                            'e':[float(frame['bbox_gt'][anomalous[frame['frame_id']]][0][2])],
                            'f':[float(frame['bbox_gt'][anomalous[frame['frame_id']]][0][3])]}
                    df=pd.DataFrame(ori_data)
                    df.to_csv('C:/Users/rosha/Desktop/tad-IROS2019-master/csv/'+str(video_name)+'.csv', mode='a', index=False, header=False)
                except:
                    ori_data={'a':[frame['frame_id']],
                          #'b':[mask_ano],
                            'c':['0'],
                            'd':['0'],
                            'e':['0'],
                            'f':['0']}
                    df=pd.DataFrame(ori_data)
                    df.to_csv('C:/Users/rosha/Desktop/tad-IROS2019-master/csv/'+str(video_name)+'.csv', mode='a', index=False, header=False)
            else:
                    ori_data={'a':[frame['frame_id']],
                          #'b':[mask_ano],
                            'c':['0'],
                            'd':['0'],
                            'e':['0'],
                            'f':['0']}
                    df=pd.DataFrame(ori_data)
                    df.to_csv('C:/Users/rosha/Desktop/tad-IROS2019-master/csv/'+str(video_name)+'.csv', mode='a', index=False, header=False)
            ##################################################################################
            ##################################################################################    
            
            if count<10:
                path="/00000"+str(count)+'.jpg'
            elif count>9 and count<100:
                path="/0000"+str(count)+'.jpg'
            elif count>99 and count<1000:
                path="/000"+str(count)+'.jpg'
                
            
            img_dir = os.path.join(image_folder + path)
            img = cv2.imread(img_dir,1)
            
           
            #print(count)
            #print(anomalous.keys())
            
            #visualize
            if count-1 in anomalous.keys():
                try:
                    
                    #draw prediction bbox(5 frame)
                    '''
                    for i in range(5):
                        start_x = (frame['bbox_pred'][anomalous[frame['frame_id']]][i][0] - frame['bbox_pred'][anomalous[frame['frame_id']]][i][2]/2)*args.W
                        start_y = (frame['bbox_pred'][anomalous[frame['frame_id']]][i][1] - frame['bbox_pred'][anomalous[frame['frame_id']]][i][3]/2)*args.H
                        end_x = (frame['bbox_pred'][anomalous[frame['frame_id']]][i][0] + frame['bbox_pred'][anomalous[frame['frame_id']]][i][2]/2)*args.W
                        end_y = (frame['bbox_pred'][anomalous[frame['frame_id']]][i][1] + frame['bbox_pred'][anomalous[frame['frame_id']]][i][3]/2)*args.H
                    
                        start_x = int(start_x.cpu().detach().numpy())
                        start_y = int(start_y.cpu().detach().numpy())
                        end_x = int(end_x.cpu().detach().numpy())
                        end_y = int(end_y.cpu().detach().numpy())

                        start = (start_x, start_y)
                        end = (end_x, end_y)

                    
                        img = cv2.rectangle(img, start, end, (0,0,255), 5)
                        #cv2.putText(img, "WARNING!", start , 0 ,1,(0,0,255),2)
                        #cv2.imshow('img',img)
                    #    temp = os.path.join("C:/Users/rosha/Desktop/tad-IROS2019-master/result/video_pred_frame/"+str(count)+str(i)+'.jpg')
                    #    cv2.imwrite(temp, img)
                    #    img = cv2.imread(img_dir,1)
                    '''
                    
                    
                    #draw gt bbox
                    start_x = (frame['bbox_gt'][anomalous[frame['frame_id']]][0][0] - frame['bbox_gt'][anomalous[frame['frame_id']]][0][2]/2)*args.W
                    start_y = (frame['bbox_gt'][anomalous[frame['frame_id']]][0][1] - frame['bbox_gt'][anomalous[frame['frame_id']]][0][3]/2)*args.H
                    end_x = (frame['bbox_gt'][anomalous[frame['frame_id']]][0][0] + frame['bbox_gt'][anomalous[frame['frame_id']]][0][2]/2)*args.W
                    end_y = (frame['bbox_gt'][anomalous[frame['frame_id']]][0][1] + frame['bbox_gt'][anomalous[frame['frame_id']]][0][3]/2)*args.H
                
                    start_x = int(start_x.cpu().detach().numpy())
                    start_y = int(start_y.cpu().detach().numpy())
                    end_x = int(end_x.cpu().detach().numpy())
                    end_y = int(end_y.cpu().detach().numpy())

                    start = (start_x, start_y)
                    end = (end_x, end_y)

                
                    img = cv2.rectangle(img, start, end, (0,0,255), 5)
                    cv2.putText(img, "WARNING!", start , 0 ,1,(0,0,255),2)
                    cv2.imshow('img',img)
                    
                    temp = os.path.join("C:/Users/rosha/Desktop/tad-IROS2019-master/result/video_pred/"+str(count)+'.jpg')
                    cv2.imwrite(temp, img)
                except:
                    cv2.imshow('img',img)
            else: 
                cv2.imshow('img',img)
                temp = os.path.join("C:/Users/rosha/Desktop/tad-IROS2019-master/result/video_pred/"+str(count)+'.jpg')
                cv2.imwrite(temp, img)
                
                
            key=cv2.waitKey(10) # ms
            if key == ord('q'):
                break
            
            count+=1
            
        if file_idx % 10 == 0:
            print(file_idx)
    
    #print(anomalous)
    #run(anomalous, seq_folder,det_file, 0.8, 1.0, 0, 0.2, None, True)

    #auc, fpr, tpr = Evaluator.compute_AUC(all_mean_iou_anomaly_scores, evaluator.labels)
    #print("FVL MEAN IOU AUC: ", auc)
    #auc, fpr, tpr = Evaluator.compute_AUC(all_mask_anomaly_scores, evaluator.labels)
    #print("FVL Mask AUC: ", auc)
    #auc, fpr, tpr = Evaluator.compute_AUC(all_min_iou_anomaly_scores, evaluator.labels)
    #print("FVL MIN IOU AUC: ", auc)
    #auc, fpr, tpr = Evaluator.compute_AUC(all_pred_std_mean_anomaly_scores, evaluator.labels)
    #print("FVL PRED STD MEAN AUC: ", auc)
    #auc, fpr, tpr = Evaluator.compute_AUC(all_pred_std_max_anomaly_scores, evaluator.labels)
    #print("FVL PRED STD MAX AUC: ", auc)


    
if __name__=='__main__':
    args = parse_args()
    main(args)