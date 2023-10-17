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
from matplotlib.pyplot import draw
import math
from scipy.stats import entropy
import Analyzer as an

class FileUpload(object):
 
    def __init__(self):
        self.fileTypes = ["png", "jpg"]
 
    def run(self):
        st.title('Chest X-ray analysis tool')
        
        file = st.sidebar.file_uploader("Upload file", type=self.fileTypes)
        
        
        st.sidebar.title("Information")
        CTR_placeholder = st.sidebar.empty()
        CTR_placeholder = st.sidebar.text("CTR: ")
        lung_placeholder = st.sidebar.empty()
        lung_placeholder = st.sidebar.text("Lung ratio: ")
        
        show_file = st.empty()
        if not file:
            show_file.info("Please upload a Chest X-ray of type: " + ", ".join(["png", "jpg"]))
            
        col1, col2 = st.columns(2)
        with col1:
            placeholder1 = st.image((np.zeros((512,512))))
            
        with col2:
            placeholder2 = st.image((np.zeros((512,512))))
            
        if file:
            show_file.info("Please start analyzing the Chest X-ray")
            
            img = Image.open(file)
            img = np.array(img)

            img = xrv.datasets.normalize(img, 255) # convert 8-bit image to [-1024, 1024] range
            if len(np.shape(img)) > 2:
                img = img.mean(2)[None, ...] # Make single color channel
            else:
                img = img[None, ...]

            transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(512)])
            
            img = transform(img)
            
            with col1:
                fig = plt.figure()
                plt.imshow(img[0],cmap='bone')
                plt.axis("off")
                plt.style.use('dark_background')
                placeholder1.pyplot(fig)

            img = torch.from_numpy(img)


            file.close()
            
        if st.sidebar.button("Analyse Chest X-ray"):
            show_file.info("Analyzing the Chest X-ray...")
            CTR, classification, H_l, H_r, size_lung_l, size_lung_r, fig = an.analyze_x_ray(img)
            with col2:
                placeholder2.pyplot(fig)
            show_file.info("Analyzed the Chest X-ray!")
            CTR_placeholder.text(f"CTR: {CTR}")
            lung_placeholder.text(f"Lung Ratio: {size_lung_l}")
            
            st.dataframe(classification.style.highlight_max(axis=0))
 
if __name__ ==  "__main__":
    helper = FileUpload()
    helper.run()