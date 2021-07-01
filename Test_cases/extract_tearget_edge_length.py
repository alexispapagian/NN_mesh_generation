# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:37:41 2019

@author: papagian
"""


import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../network_datasets')


import pickle

from Triangulation import *
import Triangulation_with_points

import torch
import torch.optim as optim
import struct



import torch.nn as nn

from torch.autograd import Variable
from math import atan2,pow,acos
from  Neural_network import *

from torch.autograd.function import Function


import matplotlib.cm as cm
import matplotlib.colors as colors
from functools import reduce

import itertools

def loadall(filename):
    lis=[]
    with open(filename, "rb") as f:
        while True:
            try:
                lis.append(pickle.load(f,encoding="latin1"))
            except EOFError:
                break
    return lis


nb_of_edges=4
for nb_of_points in [2,4]:
    # Open polygon file
    polygons_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl'
    polygons_with_points=loadall(polygons_filepath)
    polygons_with_points=polygons_with_points[0]
    
    
    # grid score files
    grid_score_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/patch_scores/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_20_sector_qualities.pkl'
    patch_scores=loadall(grid_score_filepath)
    patch_scores=patch_scores[0]
    
    
    
    # seperate into contour and points
    polygons=polygons_with_points[:,:2*nb_of_edges]
    inner_points=polygons_with_points[:,2*nb_of_edges:]
    
    
    polygon_with_target_edge_length=[]
    grid_score=[]
    # for each polygon go through target edge edge length 0.3-1
    for index,polygon in enumerate(polygons):
        print(index,"out of", len(polygons))
        polygon=polygon.reshape(nb_of_edges,2)
        for list_index,target_edge_length in enumerate([.3,.4,.5,.6,.7,.8,.9,1]):
#            if list_index%2==0:
#                point_number,points_coordinates=get_extrapoints_target_length_additional(polygon,target_edge_length,algorithm='del2d')
#    #        elif list_index%3==0:
    #            point_number,points_coordinates=get_extrapoints_target_length_jupyter(polygon,target_edge_length,algorithm='del2d')
#            else :
#                point_number,points_coordinates=get_extrapoints_target_length_jupyter(polygon,target_edge_length,algorithm='del2d')
#    
#            if point_number==nb_of_points:
                polygon_with_target_edge_length.append(np.hstack([polygon.reshape(1,2*nb_of_edges),np.array(target_edge_length).reshape(1,1)]))
                grid_score.append(patch_scores[index])
            
            
    polygon_with_target_edge_length=np.array(polygon_with_target_edge_length)
    grid_score=np.array(grid_score)
    
    # check if points inserted are same with the ones inserted
    
    polygon_target_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/patch_scores/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_target_length.pkl'
    #
    with open(polygon_target_filepath,'wb') as f :
        pickle.dump(polygon_with_target_edge_length,f)
        
        
        
    grid_score_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/patch_scores/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_20_sector_qualities_total.pkl'
    
    with open(grid_score_filepath,'wb') as f:
        pickle.dump(grid_score,f)   
    
