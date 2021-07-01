
# coding: utf-8

# In[9]:


import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../network_datasets')
sys.path.insert(0, '../point_coordinates_regression')


# In[10]:



from Triangulation import *
from point_coordinates_regression import *
from Triangulation_with_points import ray_tracing

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

from mpl_toolkits import mplot3d






import numpy as np
import matplotlib.pyplot as plt

import random

import scipy
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.optimize import fmin,minimize







# In[13]:



class quadratic_bspline(interpolate.interp2d):
    ''' Iveride interp2d to include quadratic spline'''
    def __init__(self,*args,**kws):
        try:
            super(quadratic_bspline,self).__init__(*args,**kws)
        
        except ValueError:      
            kx=ky=2
            x=args[0]
            y=args[1]
            z=args[2]
            rectangular_grid = (z.size == len(x) * len(y))
            if rectangular_grid:
                self.tck = scipy.interpolate.fitpack.bisplrep(x, y, z, kx=kx, ky=ky, s=0.0)
            else:
                nx, tx, ny, ty, c, fp, ier = scipy.interpolate.dfitpack.regrid_smth(
                x, y, z, None, None, None, None,
                kx=kx, ky=ky, s=0.0)
                self.tck = (tx[:nx], ty[:ny], c[:(nx - kx - 1) * (ny - ky - 1)],
                        kx, ky)
            self.bounds_error = False
            self.fill_value = None
            self.x, self.y, self.z = [np.array(a, copy=copy) for a in (x, y, z)]

            self.x_min, self.x_max = np.amin(x), np.amax(x)
            self.y_min, self.y_max = np.amin(y), np.amax(y)



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
    
    

#def plot_grid_qualities(contour,grid_qualities,grid_points,inner_points):
#    B=list(grid_qualities.flatten())
#    cs = plt.scatter(grid_points[:,0],grid_points[:,1],c=B,cmap=cm.RdYlGn_r,vmin=min(grid_qualities.flatten()),vmax=max(grid_qualities.flatten()),s=4)
#    plot_contour(contour)
#    plt.scatter(inner_points[:,0],inner_points[:,1],marker='o',c='b',label='Point location')
#    plt.colorbar(cs)
#    plt.legend()
#    plt.show()
    
    
def batch_size_factor(n,minimum,maximum):    
    factor_set=set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    for factor in factor_set:
        if factor>minimum and factor<maximum:
            div_factor=factor
        
    return div_factor




def select_points(contour,grid_points,grid_qualities,nb_of_points,nb_of_grid_points,target_edge_length):
    
    selected_points=[]
    surrounding_points_indices_list=[]
    surrounding_points_list=[]
    grid_qualities_surrounding_list=[]
    
    grid_qualities_duplicate=grid_qualities.flatten()
    label_added=False
    for i in range(nb_of_points):
        
        minimum_index=np.argmin(grid_qualities_duplicate)
        surrounding_points_index=np.array([minimum_index+1,minimum_index-1,minimum_index+nb_of_grid_points
                        ,minimum_index-nb_of_grid_points,minimum_index-nb_of_grid_points+1,minimum_index-nb_of_grid_points-1,
                        minimum_index+nb_of_grid_points+1,minimum_index+nb_of_grid_points-1])

    
        surrounding_points_index_2_ring=np.array([minimum_index+2,minimum_index-2,
                        minimum_index+2*nb_of_grid_points,
                        minimum_index+2*nb_of_grid_points+2,minimum_index+2*nb_of_grid_points-2,
                        minimum_index+2*nb_of_grid_points+1,minimum_index+2*nb_of_grid_points-1,                      
                                              
                                              
                        minimum_index-2*nb_of_grid_points,
                        minimum_index-2*nb_of_grid_points+1,minimum_index-2*nb_of_grid_points-1, 
                        minimum_index-2*nb_of_grid_points+2, minimum_index-2*nb_of_grid_points-2 ,
                                              
                       
                        minimum_index+nb_of_grid_points+2,
                        minimum_index+nb_of_grid_points-2,
                                              
                        minimum_index-nb_of_grid_points+2,
                        minimum_index-nb_of_grid_points-2


                                                                
                ])
        surrounding_points_index_3_ring=np.array([
                        minimum_index+3,minimum_index-3,
                        
                        minimum_index+3*nb_of_grid_points,
                        minimum_index+3*nb_of_grid_points+3,minimum_index+3*nb_of_grid_points-3,
                        minimum_index+3*nb_of_grid_points+2,minimum_index+3*nb_of_grid_points-2,
                        minimum_index+3*nb_of_grid_points+1,minimum_index+3*nb_of_grid_points-1, 
                        
                                              
                                              
                        minimum_index-3*nb_of_grid_points,
                        minimum_index-3*nb_of_grid_points+3, minimum_index-3*nb_of_grid_points-3 ,
                        minimum_index-3*nb_of_grid_points+2, minimum_index-3*nb_of_grid_points-2 ,
                        minimum_index-3*nb_of_grid_points+1,minimum_index-3*nb_of_grid_points-1, 
                       
                                              
                       
                        minimum_index+nb_of_grid_points+3,
                        minimum_index+nb_of_grid_points-3,
                                              
                        minimum_index-nb_of_grid_points+3,
                        minimum_index-nb_of_grid_points-3,
                
                         minimum_index+2*nb_of_grid_points+3,
                        minimum_index+2*nb_of_grid_points-3,
                                              
                        minimum_index-2*nb_of_grid_points+3,
                        minimum_index-2*nb_of_grid_points-3



                                                                
                ])
        
        
        
        try:
            surrounding_points=grid_points[np.asarray(surrounding_points_index)]
            surrounding_points_2_ring=grid_points[np.asarray(surrounding_points_index_2_ring)]
            surrounding_points_3_ring=grid_points[np.asarray(surrounding_points_index_3_ring)]
        except IndexError as e:
            print(e)
        
        point_maximum=grid_points[minimum_index]
        #if not label_added:
         #   plt.scatter(point_maximum[0],point_maximum[1],marker='d',s=30,c='y',label='Predictions')
          #  label_added=True
        #plt.scatter(point_maximum[0],point_maximum[1],marker='d',s=30,c='y')
        selected_points.append(np.array(point_maximum))
        
        if .6<target_edge_length<=1:
            ring=3
        elif .4<target_edge_length<.6:
            ring=2
        else:
            ring=1
        
        grid_qualities_duplicate[minimum_index]=100        
        if ring==3:
            grid_qualities_duplicate[np.asarray(surrounding_points_index)]=100
            grid_qualities_duplicate[np.asarray(surrounding_points_index_2_ring)]=100
            grid_qualities_duplicate[np.asarray(surrounding_points_index_3_ring)]=100
            surrounding_points_index=np.append(surrounding_points_index_2_ring,np.append(surrounding_points_index,minimum_index))
            surrounding_points_index=np.append(surrounding_points_index,surrounding_points_index_3_ring)
            surrounding_points_list.append(surrounding_points_3_ring,(np.append(surrounding_points_2_ring,np.append(surrounding_points,point_maximum)))),grid_qualities_surrounding_list.append(grid_qualities.flatten()[np.asarray(surrounding_points_index)]  )          
        if ring==2:
            grid_qualities_duplicate[np.asarray(surrounding_points_index)]=100
            grid_qualities_duplicate[np.asarray(surrounding_points_index_2_ring)]=100
            surrounding_points_index=np.append(surrounding_points_index_2_ring,np.append(surrounding_points_index,minimum_index))
            surrounding_points_list.append(surrounding_points_2_ring)
            surrounding_points_list.append(np.append(surrounding_points,point_maximum))
            surrounding_points_list.append(grid_qualities_surrounding_list.append(grid_qualities.flatten()[np.asarray(surrounding_points_index)]))
        else:
            surrounding_points_index=np.append(surrounding_points_index,minimum_index)
            surrounding_points_indices_list.append(surrounding_points_index)
            surrounding_points=np.append(surrounding_points,point_maximum)
            surrounding_points_list.append(surrounding_points),grid_qualities_surrounding_list.append(grid_qualities.flatten()[np.asarray(surrounding_points_index)])
            grid_qualities_duplicate[np.asarray(surrounding_points_index)]=100
#            is_inside=ray_tracing(np.array(point_maximum)[0],np.array(point_maximum)[1],contour)
#            if not is_inside:
#                selected_points.pop()
#                surrounding_points_list.pop()
#                grid_qualities_surrounding_list.pop()
#            is_inside=False
#            while not is_inside:
#               random_index,random_selected_point= random.choice(list(enumerate(selected_points)))
#               random_point_surrounding_points=np.array(surrounding_points_list[random_index])
#               random_choice=random.choice(random_point_surrounding_points.reshape(-1,2)).tolist()
#               is_inside=ray_tracing(np.array(random_choice)[0],np.array(random_choice)[1],contour)
#               if is_inside:
#                   is_inside=True
#                   selected_points.append(np.array(random_choice))
##                 
#                
                
                

    return np.array(selected_points),np.array(surrounding_points_list),np.array(grid_qualities_surrounding_list)


def select_points_updated(contour,grid_points,grid_qualities,nb_of_points,nb_of_grid_points,ring=1):
    
    selected_points=[]
    surrounding_points_indices_list=[]
    surrounding_points_list=[]
    grid_qualities_surrounding_list=[]
    
    grid_qualities_duplicate=grid_qualities.flatten()
    label_added=False
    for i in range(nb_of_points):
        
        minimum_index=np.argmin(grid_qualities_duplicate)
        
        point_maximum=grid_points[minimum_index]

        is_inside=ray_tracing(np.array(point_maximum)[0],np.array(point_maximum)[1],contour)
        look_index=len(selected_points)            

        while not is_inside:
            minimum_index=sorted([*enumerate(grid_qualities_duplicate)], key=lambda x: x[1])[look_index][0]
            point_maximum=grid_points[minimum_index]
            is_inside=ray_tracing(np.array(point_maximum)[0],np.array(point_maximum)[1],contour)
            look_index+=1    


        selected_points.append(np.array(point_maximum))

        
        surrounding_points_index=np.array([minimum_index+1,minimum_index-1,minimum_index+nb_of_grid_points
                        ,minimum_index-nb_of_grid_points,minimum_index-nb_of_grid_points+1,minimum_index-nb_of_grid_points-1,
                        minimum_index+nb_of_grid_points+1,minimum_index+nb_of_grid_points-1])

    
        surrounding_points_index_2_ring=np.array([minimum_index+2,minimum_index-2,
                        minimum_index+2*nb_of_grid_points,
                        minimum_index+2*nb_of_grid_points+2,minimum_index+2*nb_of_grid_points-2,
                        minimum_index+2*nb_of_grid_points+1,minimum_index+2*nb_of_grid_points-1,                      
                                              
                                              
                        minimum_index-2*nb_of_grid_points,
                        minimum_index-2*nb_of_grid_points+1,minimum_index-2*nb_of_grid_points-1, 
                        minimum_index-2*nb_of_grid_points+2, minimum_index-2*nb_of_grid_points-2 ,
                                              
                       
                        minimum_index+nb_of_grid_points+2,
                        minimum_index+nb_of_grid_points-2,
                                              
                        minimum_index-nb_of_grid_points+2,
                        minimum_index-nb_of_grid_points-2


                                                                
                ])
        surrounding_points_index_3_ring=np.array([
                        minimum_index+3,minimum_index-3,
                        
                        minimum_index+3*nb_of_grid_points,
                        minimum_index+3*nb_of_grid_points+3,minimum_index+3*nb_of_grid_points-3,
                        minimum_index+3*nb_of_grid_points+2,minimum_index+3*nb_of_grid_points-2,
                        minimum_index+3*nb_of_grid_points+1,minimum_index+3*nb_of_grid_points-1, 
                        
                                              
                                              
                        minimum_index-3*nb_of_grid_points,
                        minimum_index-3*nb_of_grid_points+3, minimum_index-3*nb_of_grid_points-3 ,
                        minimum_index-3*nb_of_grid_points+2, minimum_index-3*nb_of_grid_points-2 ,
                        minimum_index-3*nb_of_grid_points+1,minimum_index-3*nb_of_grid_points-1, 
                       
                                              
                       
                        minimum_index+nb_of_grid_points+3,
                        minimum_index+nb_of_grid_points-3,
                                              
                        minimum_index-nb_of_grid_points+3,
                        minimum_index-nb_of_grid_points-3,
                
                         minimum_index+2*nb_of_grid_points+3,
                        minimum_index+2*nb_of_grid_points-3,
                                              
                        minimum_index-2*nb_of_grid_points+3,
                        minimum_index-2*nb_of_grid_points-3



                                                                
                ])
        
        
        
        try:
            surrounding_points=grid_points[np.asarray(surrounding_points_index)]
            surrounding_points_2_ring=grid_points[np.asarray(surrounding_points_index_2_ring)]
            surrounding_points_3_ring=grid_points[np.asarray(surrounding_points_index_3_ring)]
        except IndexError as e:
            print(e)
        
        #if not label_added:
         #   plt.scatter(point_maximum[0],point_maximum[1],marker='d',s=30,c='y',label='Predictions')
          #  label_added=True
        #plt.scatter(point_maximum[0],point_maximum[1],marker='d',s=30,c='y')
        
        
        
        
        grid_qualities_duplicate[minimum_index]=100        
        if ring==3:
            grid_qualities_duplicate[np.asarray(surrounding_points_index)]=100
            grid_qualities_duplicate[np.asarray(surrounding_points_index_2_ring)]=100
            grid_qualities_duplicate[np.asarray(surrounding_points_index_3_ring)]=100
            surrounding_points_index=np.append(surrounding_points_index_2_ring,np.append(surrounding_points_index,minimum_index))
            surrounding_points_index=np.append(surrounding_points_index_3_ring,surrounding_points_index)
            surrounding_points_list.append(np.vstack([surrounding_points_3_ring, surrounding_points_2_ring ,np.append(surrounding_points,point_maximum).reshape(-1,2)]).reshape(-1))
            grid_qualities_surrounding_list.append(grid_qualities.flatten()[np.asarray(surrounding_points_index)])          
        if ring==2:
           grid_qualities_duplicate[np.asarray(surrounding_points_index)]=100
           grid_qualities_duplicate[np.asarray(surrounding_points_index_2_ring)]=100
           surrounding_points_index=np.append(surrounding_points_index_2_ring,np.append(surrounding_points_index,minimum_index))
           surrounding_points_list.append(np.vstack([surrounding_points_2_ring,np.append(surrounding_points,point_maximum).reshape(-1,2)]).reshape(-1))
           grid_qualities_surrounding_list.append(grid_qualities.flatten()[np.asarray(surrounding_points_index)])
        else:
            surrounding_points_index=np.append(surrounding_points_index,minimum_index)
            surrounding_points_indices_list.append(surrounding_points_index)
            surrounding_points=np.append(surrounding_points,point_maximum)
            surrounding_points_list.append(surrounding_points),grid_qualities_surrounding_list.append(grid_qualities.flatten()[np.asarray(surrounding_points_index)])
            grid_qualities_duplicate[np.asarray(surrounding_points_index)]=100
            
#            is_inside=ray_tracing(np.array(point_maximum)[0],np.array(point_maximum)[1],contour)
#            if not is_inside:
#                selected_points.pop()
#                surrounding_points_list.pop()
#                grid_qualities_surrounding_list.pop()
#            is_inside=False
#            while not is_inside:
#               random_index,random_selected_point= random.choice(list(enumerate(selected_points)))
#               random_point_surrounding_points=np.array(surrounding_points_list[random_index])
#               random_choice=random.choice(random_point_surrounding_points.reshape(-1,2)).tolist()
#               is_inside=ray_tracing(np.array(random_choice)[0],np.array(random_choice)[1],contour)
#               if is_inside:
#                   is_inside=True
#                   selected_points.append(np.array(random_choice))
#                 
#                
                
                

    return np.array(selected_points),np.array(surrounding_points_list),np.array(grid_qualities_surrounding_list)



        
def bilineaire_interpolation(surrounding_points,grid_qualities_surrounding,selected_point):
    size=int(int(len(surrounding_points))/2)
    surrounding_points=surrounding_points.reshape(size,2)
    
 
    
    z= grid_qualities_surrounding.reshape(int(sqrt(size)),int(sqrt(size)))
    if size==9:
        z_interp = quadratic_bspline(surrounding_points[:,0].reshape(int(sqrt(size)),int(sqrt(size))),surrounding_points[:,1].reshape(int(sqrt(size)),int(sqrt(size))),z, kind='quadratic')
    elif size==25:
        z_interp = quadratic_bspline(surrounding_points[:,0].reshape(int(sqrt(size)),int(sqrt(size))),surrounding_points[:,1].reshape(int(sqrt(size)),int(sqrt(size))),z, kind='quadratic')
    else :
        z_interp = interpolate.interp2d(surrounding_points[:,0].reshape(int(sqrt(size)),int(sqrt(size))),surrounding_points[:,1].reshape(int(sqrt(size)),int(sqrt(size))),z, kind='quintic')
    x_new=np.linspace(min(surrounding_points[:,0]),max(surrounding_points[:,0]),100)
    y_new=np.linspace(min(surrounding_points[:,1]),max(surrounding_points[:,1]),100)
    z_new=z_interp(x_new,y_new)
    epsilon=1e-4
    bnds=((min(surrounding_points[:,0]),max(surrounding_points[:,0])),(min(surrounding_points[:,1]),max(surrounding_points[:,1])))
    minimum=minimize(lambda v: z_interp(v[0],v[1]), np.array([selected_point[0]+epsilon,selected_point[1]+epsilon]), method='TNC',bounds=bnds)
    return np.array([minimum.x[0],minimum.x[1]])
	


def load_grid_sector_qualities(filename):
    path=os.path.join('../polygon_datasets/polygon_sector_qualities/',filename)
        
    with open(path,'rb') as input:
        grid_qualities=pickle.load(input)        
    return grid_qualities


        
def load_grid_patch_NN(filename):
    path=os.path.join('../network_datasets/grid_patch_NN',filename)
        
    with open(path,'rb') as input:
        net=pickle.load(input)        
    net.eval()
    return net


