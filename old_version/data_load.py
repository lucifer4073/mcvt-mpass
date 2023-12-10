import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import sys

from utils.Detector import Detector
from utils.properties import *
from utils.detect_time import Timer
from utils.make_video import VideoEncoder
from utils.Dataset import FrameExtractor
from utils.fps_conversion import convert_to_5fps
from utils.ROI_drawing import select_roi_from_video
import os
import re

class DataManager:
    def __init__(self,dir,qnames,gnames):
        self.qnames=qnames
        self.gnames=gnames
        self.qpaths=[os.path.join(dir,name)
                     for name in self.qnames]
        self.gpaths=[os.path.join(dir,name)
                     for name in self.gnames]
        self.qarray=[np.array(Image.open(img_path)) 
                     for img_path in self.qpaths]
        self.garray=[np.array(Image.open(img_path)) 
                     for img_path in self.gpaths]
    def get_ids(self):
        qcamids,qpids,qfnum=[],[],[]
        gcamids,gpids,gfnum=[],[],[]
        for qname in self.qpaths:
            cid,pid,fnum=extract_ids(self.qnames)
            qcamids.append(cid)
            qpids.append(pid)
            qfnum.append(fnum)
        for gname in self.gpaths:
            cid,pid,fnum=extract_ids(self.gnames)
            gcamids.append(cid)
            gpids.append(pid)
            gfnum.append(fnum)
        query=(qcamids,qpids,qfnum)
        gallery=(gcamids,gpids,gfnum)
        return query,gallery


def get_ids_images(directory): #returns images array and ids array
    fnames=os.listdir(directory)
    camid,pid,fnum=[],[],[]
    fpaths=[os.path.join(directory,fname) for fname in fnames]
    farray=[np.array(Image.open(fpath)) for fpath in fpaths]
    camids,pids,fnums=[],[],[]
    for fname in fnames:
            cid,pid,fnum=extract_ids(fname)
            camids.append(cid)
            pids.append(pid)
            fnums.append(fnum)
    return farray,np.array(pids),np.array(camids),fnums

