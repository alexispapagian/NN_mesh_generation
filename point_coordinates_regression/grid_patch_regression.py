#!/usr/bin/env python
# coding: utf-8

# In[1]:



import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../network_datasets')


# In[2]:



from Triangulation import *

import torch
import torch.optim as optim
import struct



import torch.nn as nn

from torch.autograd import Variable
from math import atan2,pow,acos
from  Neural_network import *

from torch.autograd.function import Function
from point_coordinates_regression import *



import matplotlib.cm as cm
import matplotlib.colors as colors
from functools import reduce






# In[3]:


def get_grid_qualities(XX,YY,inner_points):
    qualities=np.empty([len(XX[0,:]),len(YY[:,0])])
    for index_x,x in enumerate(XX[0,:]):
        for index_y,y in enumerate(YY[:,0]):
            quality=np.min(np.array([np.linalg.norm(point-np.array([x,y]),2) for point in inner_points]))
            qualities[index_x][index_y]=quality
    return qualities

def get_grid_qualities_with_penalty(XX,YY,inner_points,contour):
    qualities=np.empty([len(XX[0,:]),len(YY[:,0])])
    for index_x,x in enumerate(XX[0,:]):
        for index_y,y in enumerate(YY[:,0]):
            quality=np.min(np.array([np.linalg.norm(point-np.array([x,y]),2) for point in inner_points]))
            for contour_point in contour:
                if np.linalg.norm(np.array([x,y])-contour_point)<0.2:
                    quality=2
                
            qualities[index_x][index_y]=quality
    return qualities


def get_grid_qualities_with_penalty_midpoint_included(XX,YY,inner_points,contour):
    qualities=np.empty([len(XX[0,:]),len(YY[:,0])])
    for index_x,x in enumerate(XX[0,:]):
        for index_y,y in enumerate(YY[:,0]):
            quality=np.min(np.array([np.linalg.norm(point-np.array([x,y]),2) for point in inner_points]))
            for index,contour_point in enumerate(contour):
                if np.linalg.norm(np.array([x,y])-contour_point)<0.13:
                    quality=1.3
                mid_point=0.5*(contour[index]+contour[(index+1)%len(contour)])
                if np.linalg.norm(np.array([x,y])-mid_point)<0.13:
                    quality=1.3


                
            qualities[index_x][index_y]=quality
    return qualities


def plot_grid_qualities(contour,grid_qualities,grid_points,inner_points):
    plt.clf()
    B=list(grid_qualities.flatten())
    cs = plt.scatter(grid_points[:,0],grid_points[:,1],c=B,cmap=cm.RdYlGn_r,vmin=min(grid_qualities.flatten()),vmax=max(grid_qualities.flatten()),s=4)
    plot_contour(contour)
    plot_contour(contour)
    plt.scatter(inner_points[:,0],inner_points[:,1],marker='o',c='b',label='Point location')
    plt.colorbar(cs)
    plt.legend()
    plt.show()
    

def seperate_to_sectors(grid_points,nb_sectors,nb_of_grid_points):
    grid_step_size=int(nb_of_grid_points/nb_sectors)
    sectors=[] 
    indices=[]
    for q in range(nb_sectors):
        for k in range(nb_sectors):
            sector_points=[]
            for j in range(grid_step_size):
                for i in range(grid_step_size):
                    index=(grid_step_size*k+i+grid_step_size*nb_sectors*j)+(grid_step_size**2)*nb_sectors*q
                    sector_points.append(grid_points[index])
                    indices.append(index)
            sectors.append(sector_points) 
    return np.array(sectors),np.array(indices)


def get_qualities_by_sector(grid_points,inner_points,contour,nb_sectors):
    grid_step_size=int(nb_of_grid_points/nb_sectors)
    quality_sectors=[] 
    for q in range(nb_sectors):
        for k in range(nb_sectors):
            quality_point=[]
            for j in range(grid_step_size):
                for i in range(grid_step_size):
                    index=(grid_step_size*k+i+grid_step_size*nb_sectors*j)+(grid_step_size**2)*nb_sectors*q
                    grid_point=grid_points[index]
                    quality=np.min(np.array([np.linalg.norm(point-grid_point,2) for point in inner_points]))
                    for index,contour_point in enumerate(contour):
                        if np.linalg.norm(grid_point-contour_point)<0.13:
                            quality=1.3
                        mid_point=0.5*(contour[index]+contour[(index+1)%len(contour)])
                        if np.linalg.norm(grid_point-mid_point)<0.13:
                            quality=1.3
                    quality_point.append(quality)
            quality_sectors.append(quality_point) 
    return np.array(quality_sectors)
    
    

def plot_grid_qualities(contour,grid_qualities,grid_points,inner_points):
    B=list(grid_qualities.flatten())
    cs = plt.scatter(grid_points[:,0],grid_points[:,1],c=B,cmap=cm.RdYlGn_r,vmin=min(grid_qualities.flatten()),vmax=max(grid_qualities.flatten()),s=4)
    plot_contour(contour)
    plot_contour(contour)
    plt.scatter(inner_points[:,0],inner_points[:,1],marker='o',c='b',label='Point location')
    plt.colorbar(cs)
  #  plt.legend()
   # plt.show()
    
    
def batch_size_factor(n,minimum,maximum):    
    factor_set=set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    for factor in factor_set:
        if factor>minimum and factor<maximum:
            div_factor=factor
        
    return div_factor




def save_grid_sector_qualities(filename,qualities):
    path=os.path.join('../polygon_datasets/polygon_sector_qualities/',filename)

    with open(path,'wb') as output:
        pickle.dump(qualities,output)
        
def load_grid_sector_qualities(filename):
    path=os.path.join('../polygon_datasets/polygon_sector_qualities/',filename)
        
    with open(path,'rb') as input:
        grid_qualities=pickle.load(input)        
    return grid_qualities
    
    
    

def save_grid_patch_NN(filename,net):
    path=os.path.join('../network_datasets/grid_patch_NN',filename)

    with open(path,'wb') as output:
        pickle.dump(net,output)
        
def load_grid_patch_NN(filename):
    path=os.path.join('../network_datasets/grid_patch_NN',filename)
        
    with open(path,'rb') as input:
        net=pickle.load(input)        
    net.eval()
    return net

