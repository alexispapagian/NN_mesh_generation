# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:50:29 2019

@author: papagian
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:22:40 2019

@author: papagian
"""

import pickle

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:11:46 2019

@author: papagian
"""


import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../network_datasets')
import time
from contextlib import contextmanager

sys.path.insert(0, '../point_coordinates_regression/')
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, './pyDelaunay2D/')
sys.path.insert(0, '../network_datasets/')
from  grid_patch_regression import seperate_to_sectors
from bilinear_interpolation_grid import *

from Triangulation import *
import Triangulation_with_points

import torch
import torch.optim as optim
import struct
import threading
import _thread
import signal
import torch.nn as nn

from torch.autograd import Variable
from math import atan2,pow,acos
from  Neural_network import *

from torch.autograd.function import Function
from point_coordinates_regression import *
from bilinear_interpolation_grid import select_points


import matplotlib.cm as cm
import matplotlib.colors as colors
from functools import reduce

import itertools
from torch.utils.data import Dataset,DataLoader

class alt_2d_conv_net(nn.Module):
    
    def __init__(self,nb_of_filters,nb_of_hidden_nodes,out_dimension,nb_of_edges,nb_of_points):
        super(alt_2d_conv_net,self).__init__()
        
        self.nb_of_edges=nb_of_edges
        self.nb_of_points=nb_of_points

        self.nb_of_filters=nb_of_filters
        
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=nb_of_filters,stride=1,kernel_size=(2,1)),
                                 nn.MaxPool2d(stride=1,kernel_size=(2,1)),nn.ReLU(inplace=True))
        self.conv2=nn.Sequential(nn.Conv2d(in_channels=1,out_channels=nb_of_filters,stride=1,kernel_size=(2,1)),
                                 nn.MaxPool2d(stride=1,kernel_size=(2,1)),nn.ReLU(inplace=True))
           
    

        self.fc=nn.Sequential(  nn.BatchNorm1d(num_features=nb_of_filters*2*(nb_of_edges-1)+2*(nb_of_points)),

                                nn.Linear(2*nb_of_filters*(nb_of_edges-1)+2*(nb_of_points),nb_of_hidden_nodes),
                                        nn.Tanh(),
                               nn.BatchNorm1d(num_features=nb_of_hidden_nodes),
                              
                              nn.Linear(nb_of_hidden_nodes,nb_of_hidden_nodes),
                                        nn.Tanh(),
                               nn.BatchNorm1d(num_features=nb_of_hidden_nodes),
                               
                              nn.Linear(nb_of_hidden_nodes,nb_of_hidden_nodes),
                                        nn.Tanh(),
                               nn.BatchNorm1d(num_features=nb_of_hidden_nodes),
                    
                              

                               nn.Linear(nb_of_hidden_nodes,out_dimension) )                                                                        
        
        
        
        
    def forward(self,x):
        
        polygons_points=x.narrow(1,0,1).narrow(2,0,self.nb_of_edges+1)
        inner_points=x.narrow(1,0,1).narrow(2,nb_of_edges+1,nb_of_points).resize(len(x),2*self.nb_of_points)
        
       
        conv_result1=self.conv1(polygons_points)
        #conv_result2=self.conv2(inner_points)        
        
        # reshape the convolution results
        
        conv_result1=conv_result1.view(-1,self.nb_of_filters*(2*(self.nb_of_edges-1)))
        #conv_result2=conv_result2.view(-1,self.nb_of_filters*(2*(self.nb_of_points-1)))
    
        concat_tensor=torch.cat([conv_result1,inner_points],1)
        output=self.fc(concat_tensor)
        return output



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


    nb_of_edges=6
    nb_of_points=1
    
    
    
    validation_dataset_filepath="validation_datasets/"+str(nb_of_edges)+"_"+str(nb_of_points)+"_validation_set"
    polygons_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl'
    network_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/networks/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_'+str(nb_of_grid_points)+'_grid_NN'
    target_edge_length=0.5
    
    # load polygon data 
    polygons_with_points=loadall(polygons_filepath)
    polygons_with_points=polygons_with_points[0]
    
    
    # load network
    with open(network_filepath,'rb') as f:
        NN2=pickle.load(f)
    
    nb_of_samples=50
    
    if os.path.exists(validation_dataset_filepath):
        with open(validation_dataset_filepath,"rb") as f:
            validation_polygon_set=pickle.load(f)
    else:
        validation_polygon_set=[]
        for i in range(nb_of_samples):
                notfound=True    
                while notfound:
                    polygon=apply_procrustes(generate_contour(nb_of_edges))
                    point_number,point_coordinates=get_extrapoints_target_length_additional(polygon,0.5,algorithm='del2d')
                    if point_number==nb_of_points:
                        notfound=False
                        polygon_with_points=np.vstack([polygon,point_coordinates])
                        validation_polygon_set.append(polygon_with_points)
        with open(validation_dataset_filepath,"wb") as f:
                pickle.dump(validation_polygon_set,f)
    
    triangulation_errors=[]
    triangulation_errors_mean=[]
    for polygon_with_points in validation_polygon_set:
        # select a random contour
    #    polygon_with_points=polygons_with_points[46]
    #    polygon_with_points=polygon_with_points.reshape(nb_of_edges+nb_of_points,2)
        polygon=polygon_with_points[:nb_of_edges]
        inner_points=polygon_with_points[nb_of_edges:]
        
        # Triangulate using algorithm
            # Calculate the connection table
        real_qualities,_=Triangulation_with_points.quality_matrix(polygon,inner_points)
        real_qualities=np.array(real_qualities)
            # Order the connection table
        real_ordered_qualities=Triangulation_with_points.order_quality_matrix(real_qualities,polygon,np.vstack([polygon,inner_points]),check_for_equal=True)
        initial_elements,sub_elements=Triangulation_with_points.triangulate(polygon,inner_points,real_ordered_qualities,recursive=True,plot_mesh=False)
        total_elements=concat_element_list(initial_elements,sub_elements)
        mesh_minimum_quality=compute_minimum_quality_triangulated_contour(polygon_with_points,total_elements)
        mesh_mean_quality=compute_mean_quality_triangulated_contour(polygon_with_points,total_elements)

        
            
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
        predicted_points,surrounding_points_list,grid_qualities_surrounding=select_points_updated(polygon,grid_points,grid_qualities,nb_of_points,nb_of_grid_points,1)
        
        
##        # Interpolate
        predicted_points=[point  for i in range(nb_of_points) for point in bilineaire_interpolation(surrounding_points_list[i],grid_qualities_surrounding[i],predicted_points[i])]
        predicted_points=np.array(predicted_points).reshape(nb_of_points,2)
        predicted_points=np.unique(predicted_points,axis=0)
        if len(predicted_points)<nb_of_points:
            continue
        
#        
#        
               
        # Triangulate with new points
        qualities_with_predicted_points,_=Triangulation_with_points.quality_matrix(polygon,predicted_points)
        ordered_qualities_with_predicted_points=Triangulation_with_points.order_quality_matrix(qualities_with_predicted_points,polygon,np.vstack([polygon,predicted_points]),check_for_equal=False)
#        
#        initial_elements,sub_elements=Triangulation_with_points.triangulate(polygon,predicted_points,ordered_qualities_with_predicted_points,recursive=True,plot_mesh=True)
#        total_elements=concat_element_list(initial_elements,sub_elements)
#        mesh_minimum_quality_with_predicted_points=compute_minimum_quality_triangulated_contour(np.vstack([polygon,predicted_points]),total_elements)
#        mesh_mean_quality_with_predicted_points=compute_mean_quality_triangulated_contour(np.vstack([polygon,predicted_points]),total_elements)

        # Compute Triangulation error    
       
        try:
            with time_limit(20):
                initial_elements,sub_elements=Triangulation_with_points.triangulate(polygon,predicted_points,ordered_qualities_with_predicted_points,recursive=True,plot_mesh=True)
                total_elements=concat_element_list(initial_elements,sub_elements)
                mesh_minimum_quality_NN=compute_minimum_quality_triangulated_contour(np.vstack([polygon,predicted_points]),total_elements)
                mesh_mean_quality_NN=compute_mean_quality_triangulated_contour(np.vstack([polygon,predicted_points]),total_elements)

                triangulation_error=100*(mesh_minimum_quality-mesh_minimum_quality_NN)/mesh_minimum_quality
                triangulation_error_mean=100*(mesh_mean_quality-mesh_mean_quality_NN)/mesh_mean_quality

                triangulation_errors.append(triangulation_error)
                triangulation_errors_mean.append(triangulation_error_mean)
        except TimeoutException as e:
            print("Timed out!")
            continue

    triangulation_errors=np.array(triangulation_errors)
    triangulation_errors_mean=np.array(triangulation_errors_mean)

    print("Average min triangulation error using NN2 for {} edge polygons with {} inner points is {}% ".format(nb_of_edges,nb_of_points,triangulation_errors.mean()))
    print("Average mean triangulation error using NN2 for {} edge polygons with {} inner points is {}% ".format(nb_of_edges,nb_of_points,triangulation_errors_mean.mean()))