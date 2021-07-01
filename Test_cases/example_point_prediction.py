
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:22:40 2019

@author: papagian
"""

import pickle




import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../network_datasets')
sys.path.insert(0, '../point_coordinates_regression/')

import time
from contextlib import contextmanager

from  grid_patch_regression import seperate_to_sectors
from bilinear_interpolation_grid import *

from Triangulation import *
from IPython import get_ipython

import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
from math import atan2,pow,acos
from  Neural_network import *

from torch.autograd.function import Function
from point_coordinates_regression import *
from bilinear_interpolation_grid import select_points_updated


import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader


get_ipython().run_line_magic('matplotlib', 'qt')



def loadall(filename):
    lis=[]
    with open(filename, "rb") as f:
        while True:
            try:
                lis.append(pickle.load(f))
            except EOFError:
                break
    return lis

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()
        
        
        
if __name__=='__main__':
    
    # Define the grid    
    nb_of_grid_points=20
    
    X=np.linspace(-1.3,1.3,nb_of_grid_points)
    Y=np.linspace(-1.3,1.3,nb_of_grid_points)
    XX,YY=np.meshgrid(X,Y)
    grid_points=np.array([[x,y] for x in X for y in Y])
        
    nb_sectors=int(nb_of_grid_points/2)
    sectors,indices=seperate_to_sectors(grid_points,nb_sectors,nb_of_grid_points)
    grid_step_size=int(nb_of_grid_points/nb_sectors)


    nb_of_edges=12
    nb_of_points=14
    
    
    
    polygons_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl'
    network_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/networks/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_'+str(nb_of_grid_points)+'_grid_NN'
    
    # You can use this target edge length for one point it won't make any difference
    target_edge_length=.3
    
    # load polygon data 
    polygons_with_points=loadall(polygons_filepath)[0]
    
    
    # load network
    with open(network_filepath,'rb') as f:
        NN2=pickle.load(f)
    
    
    # Get a random polygon from dataset
    index=89
    polygon_with_point=polygons_with_points[index]
    polygon=polygon_with_point[:2*nb_of_edges].reshape(nb_of_edges,2)
    real_point=polygon_with_point[2*nb_of_edges:]

        
            
    # use network to extract predicted points
    polygon_with_target_edge_length=np.hstack([polygon.reshape(2*nb_of_edges),np.array(target_edge_length).reshape(1)])
    
    # Adding grid points of each patch for the input of the NN
    polygon_with_grid_points=[]
    for sector in sectors:
        polygon_with_sector_points=np.hstack([polygon_with_target_edge_length.reshape(1,len(polygon_with_target_edge_length)),sector.reshape(1,2*len(sector))])
        polygon_with_sector_points=Variable(torch.from_numpy(polygon_with_sector_points))
        polygon_with_sector_points=polygon_with_sector_points.expand(1,polygon_with_sector_points.shape[1]).type(torch.FloatTensor)
        polygon_with_grid_points.append(polygon_with_sector_points)

    # Evaluation mode for network and trasfer to cpu
    NN2=NN2.cpu().eval()
    
    
    # Infer  grid scores ftom NN
    sector_qualities=[]
    for polygon_with_sector_points in polygon_with_grid_points:
        sector_quality=NN2(polygon_with_sector_points)
        sector_qualities.append(sector_quality.data[0].numpy())
  
    sector_qualities=np.array(sector_qualities)


    grid_qualities=np.empty((grid_step_size**2)*(nb_sectors**2))
    for index,point_index in enumerate(indices):
        grid_qualities[point_index]=sector_qualities.flatten()[index]


    # Point selection
    predicted_points,surrounding_points_list,grid_qualities_surrounding=select_points_updated(polygon,grid_points,grid_qualities,nb_of_points,nb_of_grid_points,target_edge_length)
    
    
##  # Interpolate
    predicted_points=[point  for i in range(nb_of_points) for point in bilineaire_interpolation(surrounding_points_list[i],grid_qualities_surrounding[i],predicted_points[i])]
    predicted_points=np.array(predicted_points).reshape(nb_of_points,2)
    predicted_points=np.unique(predicted_points,axis=0)
    
    # Plotting
    plot_contour(polygon)
    plt.scatter(real_point[0],real_point[1])
    plt.scatter(predicted_points[0][0],predicted_points[0][1])