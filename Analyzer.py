#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 12:19:02 2023

@author: bartdevries
"""
from enum import Enum
from io import BytesIO, StringIO
from typing import Union
import torchxrayvision as xrv 
import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import skimage, torch, torchvision
import matplotlib.pyplot as plt
import math
from scipy.stats import entropy
def analyze_x_ray(img):
    # Load model and process image
    model = xrv.models.DenseNet(weights="densenet121-res224-all")

    "Classification of the chest x-ray"
    classification = model(img[None,...]) # or model.features(img[None,...]) 
    classification = pd.DataFrame({'Pathology': model.pathologies,'Probability':classification[0].detach().numpy()})
    print(classification)
    
    seg_model = xrv.baseline_models.chestx_det.PSPNet()
    segm = seg_model(img)
    segm =  segm.detach().numpy()
    segm_prob = segm
    
    img = img.detach().numpy()
    
    segm = 1 / (1 + np.exp(-segm))  # sigmoid
    segm[segm < 0.5] = 0
    segm[segm > 0.5] = 1
    
    heart_mask = segm[:,8,...]
    lung_l_mask = segm[:,4,...] 
    lung_r_mask =  segm[:,5,...]
    lungs_mask = segm[:,4,...] + segm[:,5,...]
    vertebrae_mask = segm[:,13,...]
    mediastinum_mask = segm[:,11,...]
    clavicula_l_mask = segm[:,0,...]
    clavicula_r_mask = segm[:,1,...]
    clavicula_mask = segm[:,0,...] + segm[:,1,...]
    trachea_mask = segm[:,12,...]
    
    segm_prob *=segm
    lungs_prob = segm_prob[0,4,...]+segm_prob[0,5,...]
    max_lungs = np.quantile(lungs_prob,q=0.95)
    rois_path = np.where(lungs_prob<max_lungs*0.75,lungs_mask[0],0)
    rois_path = np.where(rois_path==heart_mask[0],0,rois_path)
    rois_path = np.where(rois_path==clavicula_mask[0],0,rois_path)
    lungs_seg = np.where(rois_path!=0,img[0],0)
    
    lung_l_mask = np.where(lung_l_mask[0]==heart_mask[0],0,lung_l_mask[0])
    lung_l_mask = np.where(lung_l_mask==clavicula_mask[0],0,lung_l_mask)
    lung_l_mask = np.where(lung_l_mask==mediastinum_mask[0],0,lung_l_mask)
    lung_l_seg = np.where(lung_l_mask!=0,img[0],0)
    
    lung_r_mask = np.where(lung_r_mask[0]==heart_mask[0],0,lung_r_mask[0])
    lung_r_mask = np.where(lung_r_mask==clavicula_mask[0],0,lung_r_mask)
    lung_r_mask = np.where(lung_r_mask==mediastinum_mask[0],0,lung_r_mask)
    lung_r_seg = np.where(lung_r_mask!=0,img[0],0)

    
    base = 2  # work in units of bits
    H_l = entropy(lung_l_seg[lung_l_seg>0], base=base)
    H_r = entropy(lung_r_seg[lung_r_seg>0], base=base)
    
    size_lung_l = np.count_nonzero(segm[:,4,...] ==1)
    size_lung_r = np.count_nonzero(segm[:,5,...] ==1)
    
    dif_lung = int(((size_lung_l/size_lung_r)*100))
    #print(dif_lung)
    if dif_lung>110 or dif_lung<90 or H_r>14 or H_l>14:
        #print(f'Caution possible pathology in lungs - Lung size diff: {dif_lung}% - E left: {H_l} - E right: {H_r}')
        path_lung = 1
    else:
        path_lung = 0

    def pytha(mask,origin=(0,0)):
        shape = np.shape(mask)
        outputs = []
        for x in range(shape[0]):
            for y in range(shape[1]):
                if np.max(mask[y,x])==1:
                    if origin == (0,0):
                        a = origin[0] + x
                        b = origin[1] + y
                    elif origin == (512,0):
                        a = origin[0] - x
                        b = origin[1] + y
                       
                    c = math.sqrt(a ** 2 + b ** 2)
                    outputs.append((x,y,c))
        outputs = np.array(outputs)    
        idx = np.where(outputs==np.max(outputs[...,2]))
        output = outputs[idx[0]]
        return output
        
    
    l_lung = pytha(lung_l_mask,origin=(0,0))
    r_lung = pytha(lung_r_mask,origin=(512,0))
    
    def x_extrem(mask,origin_x=0):
        shape = np.shape(mask)
        if origin_x==0:
            outputs = []
            for x in range(shape[1]):
                for y in range(shape[2]):
                    if np.max(mask[:,y,x])==1:
                        outputs.append((y, x))
                if outputs:
                    outputs = np.array(outputs)
                    output = (x,int(np.median(outputs[0])))
                    print(output)
                    return output   
        
        elif origin_x==512:
            outputs = []
            for x in reversed(range(shape[1])):
                for y in range(shape[2]):
                    if np.max(mask[:,y,x])==1:
                        outputs.append((y, x))
                if outputs:
                    outputs = np.array(outputs)
                    output = (x,int(np.median(outputs[0])))        
                    print(output)
                    return output 
    
    l_heart= x_extrem(heart_mask,origin_x=512)
    r_heart = x_extrem(heart_mask,origin_x=0)
    
    def find_midline(mask):
        shape = np.shape(mask)
        r_lats = []
        l_lats = []
        midline = []
        for y in range(shape[2]):
            for x in range(shape[1]):
                if np.max(mask[:,y,x])==1:
                    r_lats.append((y,x))
                    break
        for y in range(shape[2]):
            for x in reversed(range(shape[1])):
                if np.max(mask[:,y,x])==1:
                    l_lats.append((y,x))
                    break
        
        r_lats = np.array(r_lats)
        l_lats = np.array(l_lats)
        
        for y_r, x_r in r_lats:
            for y_l, x_l in l_lats:
                #print(y_r,y_l)
                if y_r == y_l:
                    mid = int((x_l+x_r)/2)
                    midline.append((y_r,mid))
        
        midline = np.array(midline[5:-5])
        shift = int(((np.max(midline[:,1]) / 
                      np.min(midline[:,1])) * 100)-100)
        return midline, shift
    
    midline_vrt, shift_vrt = find_midline(vertebrae_mask)
    midline_tr, shift_tr = find_midline(trachea_mask)
    
    lungs_width = l_lung[0,0] - r_lung[0,0]
    heart_width = l_heart[0] - r_heart[0]
    
    CTR = int((heart_width/lungs_width)*100)
    lungs_axis = np.max((l_lung[0,1] ,r_lung[0,1]))
    
    y_heart = int((r_heart[1]+l_heart[1])/2)
    idx = np.where(midline_vrt[...,0]==y_heart)
    midline_heart = midline_vrt[idx[0]][0,1]
    
    fig = plt.figure()
    
    x = [r_heart[0]-1,r_heart[0]-1]
    y = [y_heart+50, y_heart-50]
    plt.plot(x, y, color="blue", linewidth=2)
    x = [l_heart[0]+1,l_heart[0]+1]
    plt.plot(x, y, color="red", linewidth=2)
    x_axis = [midline_heart,l_heart[0]]
    y = [y_heart, y_heart]
    plt.plot(x_axis, y, color="red", linewidth=2)
      
    x_axis = [r_heart[0],midline_heart]
    y = [y_heart, y_heart]
    plt.plot(x_axis, y, color="blue", linewidth=2)

    
    x = [r_lung[0,0]-1,r_lung[0,0]-1]
    y = [lungs_axis+75, lungs_axis-75]
    plt.plot(x, y, color="blue", linewidth=2)
    x = [l_lung[0,0]+1,l_lung[0,0]+1]
    plt.plot(x, y, color="blue", linewidth=2)
    x_axis = [r_lung[0,0],l_lung[0,0]]
    y = [lungs_axis, lungs_axis]
    plt.plot(x_axis, y, color="blue" , linewidth=2)
    
    x = midline_vrt[10:-20,1]
    y = midline_vrt[10:-20,0]
    plt.plot(x, y, color="yellow", linewidth=1, linestyle='dashed')
    
    plt.imshow(img[0,...],cmap='bone')
    plt.imshow(lungs_mask[0,...],alpha=0.2)
    plt.axis("off")
    plt.style.use('dark_background')

    return CTR, classification, H_l, H_r, size_lung_l, size_lung_r, fig
