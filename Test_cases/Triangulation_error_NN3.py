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

import threading
import _thread
from contextlib import contextmanager

import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../network_datasets')



sys.path.insert(0, '../point_coordinates_regression/')
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, './pyDelaunay2D/')
sys.path.insert(0, '../network_datasets/')
from  grid_patch_regression import seperate_to_sectors

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

    
def batch_size_factor(n,minimum,maximum):    
    factor_set=set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    for factor in factor_set:
        if factor>minimum and factor<maximum:
            div_factor=factor
        
    return div_factor


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
    nb_of_points=2
    
    
    
    validation_dataset_filepath="validation_datasets/"+str(nb_of_edges)+"_"+str(nb_of_points)+"_validation_set"
    polygons_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl'
    network_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/networks/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_'+str(nb_of_grid_points)+'_grid_NN'
    target_edge_length=0.5
    
    # load polygon data 
    polygons_with_points=loadall(polygons_filepath)
    polygons_with_points=polygons_with_points[0]
    
    nb_of_samples=10
    
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
    
    
#    polygontest_test_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl'

    polygon_with_points=polygons_with_points
    
    
    triangulation_errors=[]
    triangulation_errors_mean=[]

    for polygon_index,polygon_with_points in enumerate(validation_polygon_set):
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

        
        
         
        network_path2=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/networks/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_NN_conn_conv.pkl'
    
        
        with open(network_path2,'rb') as f:
            NN3=pickle.load(f)
            
        NN3=NN3.cpu().eval()
        
        # Reshape polygon with points for NN3 use
        polygon_with_points_reshaped=np.insert(polygon_with_points,nb_of_edges,polygon_with_points[0],axis=0)
        
        polygon_with_points_tensor=Variable(torch.from_numpy(polygon_with_points_reshaped)).resize(1,1,polygon_with_points_reshaped.shape[0],polygon_with_points_reshaped.shape[1]).type(torch.FloatTensor)
        
        
        predicted_qualities=NN3(polygon_with_points_tensor)
        
        predicted_qualities=predicted_qualities.detach().numpy().reshape(nb_of_edges,nb_of_edges+nb_of_points)
        
        #ordered_qualities_NN3=Triangulation_with_points.order_quality_matrix(predicted_qualities,polygon,np.vstack([polygon,inner_points]),check_for_equal=False)
    
        #initial_elements,sub_elements=Triangulation_with_points.triangulate(polygon,inner_points,ordered_qualities_NN3,recursive=False,plot_mesh=False)
    
        
        
         
        # Reshape polygon with points for NN3 use
        polygon_with_inner_points=np.vstack([polygon,inner_points])
        polygon_with_inner_points_reshaped=np.insert(polygon_with_inner_points,nb_of_edges,polygon_with_inner_points[0],axis=0)
        
        polygon_with_inner_points_tensor=Variable(torch.from_numpy(polygon_with_inner_points_reshaped)).resize(1,1,polygon_with_inner_points_reshaped.shape[0],polygon_with_inner_points_reshaped.shape[1]).type(torch.FloatTensor)
        
        
        predicted_qualities=NN3(polygon_with_inner_points_tensor)
        
        predicted_qualities=predicted_qualities.detach().numpy().reshape(nb_of_edges,nb_of_edges+nb_of_points)
        
        ordered_qualities_NN=Triangulation_with_points.order_quality_matrix(predicted_qualities,polygon,np.vstack([polygon,inner_points]),check_for_equal=True)
    

    
        try:
            with time_limit(20):
                initial_elements,sub_elements=Triangulation_with_points.triangulate(polygon,inner_points,ordered_qualities_NN,recursive=True,plot_mesh=True)
                total_elements=concat_element_list(initial_elements,sub_elements)
                mesh_minimum_quality_NN=compute_minimum_quality_triangulated_contour(polygon_with_inner_points,total_elements)
                mesh_mean_quality_NN=compute_mean_quality_triangulated_contour(polygon_with_inner_points,total_elements)

                triangulation_error=100*(mesh_minimum_quality-mesh_minimum_quality_NN)/mesh_minimum_quality
                triangulation_mean_error=100*(mesh_mean_quality-mesh_mean_quality_NN)/mesh_mean_quality

                triangulation_errors.append(triangulation_error)
                triangulation_errors_mean.append(triangulation_mean_error)

        except TimeoutException as e:
            print("Timed out!")
            continue
    triangulation_errors_mean=np.array(triangulation_errors_mean)
    triangulation_errors=np.array(triangulation_errors)
    print("Average triangulation error using NN3 for {} edge polygons with {} inner points is {}% ".format(nb_of_edges,nb_of_points,triangulation_errors.mean()))
    print("Average mean triangulation error using NN3 for {} edge polygons with {} inner points is {}% ".format(nb_of_edges,nb_of_points,triangulation_errors_mean.mean()))