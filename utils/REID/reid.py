from utils.REID.extract_ids import extract_ids
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
from utils.plotting_funcs import plot_heatmap
from data_load import DataManager
from Model import Runner
class treid:
    #cam-> single or double
    def __init__(self,dir,qnames,gnames,cam="single",model_name=""):
        self.qnames=qnames
        self.gnames=gnames
        self.dir=dir
        datamanager=DataManager(self.dir,self.qnames,self.gnames)
        self.query_arr=datamanager.qarray
        self.gallery_arr=datamanager.garray
        query,gallery=datamanager.get_ids()
        self.qids,self.qcids,self.qfnum=query
        self.gids,self.gcids,self.gfnum=gallery
    def distance_matrix(self):
        #Create Model
        model = Runner()
        qfeats=model._features(self.query_arr)
        gfeats=model._features(self.gallery_arr)
        dmat=model.compute_disance(qfeats,gfeats)

        return dmat
    def reid_ranking(self,dmat):
        indices=np.argmin(dmat,axis=1)
        return self.gids[indices]
    def visualize_dmat(self,dmat):
        plot_heatmap(dmat)
    
    




        
        
        