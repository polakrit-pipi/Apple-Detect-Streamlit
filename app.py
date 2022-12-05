from lobe import ImageModel
import os
import shutil
import random
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
from datetime import datetime
import time
import torch


model = ImageModel.load(r'model/')



def imageInput(device, src):
    
    if src == 'อัปโหลดรูปภาพ':
        image_file = st.file_uploader("ตรวจสอบรูปภาพ", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='รูปภาพที่นำเข้ามา', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            
            img = mpimg.imread('./{}'.format(imgpath))
            imgplot = plt.imshow(img)
            plt.axis('off')
            plt.show()
            img = Image.open('./{}'.format(imgpath))
            img = img.resize((160, 160), Image.ANTIALIAS)
            result = model.predict(img)
            # Print top prediction
            print('Model Prediction : {}'.format(result.prediction))
            # Print all classes
            for label, confidence in result.labels:
                print(f"{label}: {round(confidence*100,2)}%")
                pp= label
                if (pp == "HEALTHY LEAVES"):
                    st.write(f"ใบที่มีสุขภาพดี: {round(confidence*100,2)}%")
                elif (pp == "APPLE ROT LEAVES"):
                    st.write(f"โรคใบเน่า: {round(confidence*100,2)}%")
                elif (pp == "SCAB LEAVES"):
                    st.write(f"โรคใบจุด: {round(confidence*100,2)}%")
                elif (pp == "LEAF BLOTCH"):
                    st.write(f"โรคใบไหม้: {round(confidence*100,2)}%")
                    
            #--Display predicton
            #++++++++++++++++++++++++++++++++++++++
            # img_ = Image.open(outputpath)
            # with col2:
            #     st.image(img_, caption='ผลลัพธ์จากการตรวจสอบ', use_column_width='always')


def main():
    # -- Sidebar
    st.sidebar.title('🌿 Ai Health Check Apple Leaf')
    datasrc = st.sidebar.radio("เลือกประเภทรูปแบบการนำเข้า", ['อัปโหลดรูปภาพ'])
    
        
                
    option = st.sidebar.radio("ระบุประเภทข้อมูล", ['Image'], disabled = False)
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("ประมวลผลโดยใช้", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("ประมวลผลโดยใช้", ['cpu', 'cuda'], disabled = True, index=0)
    # -- End of Sidebar

    st.header('Ai Health Check Apple Leaf')
    
    if option == "Image":    
        imageInput(deviceoption, datasrc)

if __name__ == '__main__':
  
    main()
       