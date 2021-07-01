# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:30:50 2019

@author: papagian
"""
import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')
sys.path.insert(0, '../network_datasets/')
from functools import reduce

from Triangulation import *
import pickle,os
import torch
import torch.optim as optim

import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from Neural_network import Net
from point_coordinates_regression import *


edges=[4,5,6,7,8,9,10,11,12,13,14]

for nb_of_edges in edges:
    filedirectory='..//polygon_datasets//'+str(int(nb_of_edges))+'_polygons_with_points'
    del_points=load_dataset(str(nb_of_edges)+'_point_coordinates_del.pkl')
    number_of_insertion_points=load_dataset(str(nb_of_edges)+'_nb_of_points_del.pkl')
    
    
    set_points=get_set_nb_of_points(del_points)
    
    for nb_of_inner_points in set_points:
        if nb_of_inner_points==0:
            continue
        
        print(nb_of_inner_points)
        polygons_filename=os.path.join(filedirectory, str(nb_of_edges)+'_'+str(nb_of_inner_points)+'_polygons_with_points')
        qualities_filename=os.path.join(filedirectory,str(nb_of_edges)+'_'+str(nb_of_inner_points)+'_polygons_qualities_min.pkl')
        
        with open (polygons_filename,'rb') as file:
            polygons_with_points=pickle.load(file)
            
            
        
        with open (qualities_filename,'rb') as file:
            quality_matrices=pickle.load(file)
        
        if (nb_of_edges==8 and nb_of_inner_points==1) or (nb_of_edges==10 and nb_of_inner_points==1) or (nb_of_edges==12 and nb_of_inner_points==4):
            filename_polygons='../polygon_datasets/additional_polygon_datasets/'+str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_inner_points)+'_polygons_with_points.pkl'
            filename_min_qualities='../polygon_datasets/additional_polygon_datasets/'+str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_inner_points)+'_min_qualities.pkl'
         #   filename_polygons2='../polygon_datasets/additional_polygon_datasets/'+str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_inner_points)+'_polygons_with_points_part_2.pkl'
          #  filename_min_qualities2='../polygon_datasets/additional_polygon_datasets/'+str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_inner_points)+'_min_qualities_part_2.pkl'
            with open(filename_polygons,'rb') as f:
                additional_polygons_with_points=pickle.load(f)
            with open(filename_min_qualities,'rb') as f:
                additional_min_qualities=pickle.load(f)  
          #  with open(filename_polygons2,'rb') as f:
           #     additional_polygons_with_points2=pickle.load(f)
            #with open(filename_min_qualities2,'rb') as f:
             #   additional_min_qualities2=pickle.load(f)  
            print("adding datasets")    
            polygons_with_points=np.vstack([polygons_with_points,additional_polygons_with_points])
           # polygons_with_points=np.vstack([polygons_with_points,additional_polygons_with_points2])
            quality_matrices=np.vstack([quality_matrices,additional_min_qualities])
            #quality_matrices=np.vstack([quality_matrices,additional_min_qualities2])

        
        polygons=polygons_with_points[:,:2*nb_of_edges]
        inner_points=polygons_with_points[:,2*nb_of_edges:]
        
        # Testing data are usually 20 percent of the whole
        
        nb_test_data=int(0.1*polygons_with_points.shape[0])
        nb_training_data=int(polygons_with_points.shape[0])-nb_test_data
        
        
        if nb_training_data<2:
            continue
        
        quality_matrices=np.array(quality_matrices)
        polygons_with_points,quality_matrices=unison_shuffled_copies(polygons_with_points,quality_matrices)
        
        # Organizing data 
        training_data = polygons_with_points[:nb_training_data]
        testing_data  = polygons_with_points[nb_training_data:]
        
        training_labels=np.array(quality_matrices[:nb_training_data])
        testing_labels=np.array(quality_matrices[nb_training_data:])
        
        # Convert to pytorch tennsors
        
        x_tensor=torch.from_numpy(training_data).type(torch.FloatTensor)
        x_tensor_test=torch.from_numpy(testing_data).type(torch.FloatTensor)
        
        
        y_tensor=torch.from_numpy(training_labels).type(torch.FloatTensor)
        y_tensor_test=torch.from_numpy(testing_labels).type(torch.FloatTensor)
        
        
        # Convert to pytorch variables
        x_variable=Variable(x_tensor)
        x_variable_test=Variable(x_tensor_test)
        
        
        
        y_variable=Variable(y_tensor)
        y_variable=y_variable.resize(nb_training_data,1,nb_of_edges*(nb_of_edges+nb_of_inner_points))
        
        
        y_variable_test=Variable(y_tensor_test)
        y_variable_test=y_variable_test.resize(nb_test_data,1,nb_of_edges*(nb_of_edges+nb_of_inner_points))
        
        
        
        my_net=Net(polygons_with_points.shape[1],nb_of_edges*(nb_of_edges+nb_of_inner_points),nb_of_hidden_layers=2,
                   nb_of_hidden_nodes=(2*nb_of_edges + 2*nb_of_inner_points)+int(0.5*(2*nb_of_edges + 2*nb_of_inner_points)),batch_normalization=True)
        
        
        optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-4,weight_decay=.2)
        #optimizer = torch.optim.SGD(my_net.parameters(), lr=1e-5,weight_decay=.5,momentum=0.9)
        
        loss_func = torch.nn.MSELoss(size_average=False)  
        
        
#        if  torch.cuda.is_available():
#            my_net.cuda()
#            loss_func.cuda()
#            x_variable , y_variable = x_variable.cuda(), y_variable.cuda()
#            x_variable_test , y_variable_test = x_variable_test.cuda(), y_variable_test.cuda()
#            print("cuda activated")
        nb_of_epochs=10
        batch_size=int(x_variable.shape[0])
        print(len(training_data),len(testing_data))
        # Train the network #
        for t in range(nb_of_epochs):
            sum_loss=0
            for b in range(0,x_variable.size(0),batch_size):
                out = my_net(x_variable.narrow(0,b,batch_size))                 # input x and predict based on x
                loss = loss_func(out, y_variable.narrow(0,b,batch_size))     # must be (1. nn output, 2. target), the target label is NOT one-hotted
                sum_loss+=loss.item()
                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                #print(t,loss.data[0])
                optimizer.step()        # apply gradients
            
            my_net.eval()
            test_loss=loss_func(my_net(x_variable_test),y_variable_test).item()
            my_net.train()
           # print("Epoch:",t,"Training Loss:",sum_loss/x_variable.size(0),"Test Loss:",test_loss/x_variable_test.size(0))
        
        home_directory='..//network_datasets//connectivity_NN//'
        filename=os.path.join(home_directory,str(nb_of_edges)+'_'+str(nb_of_inner_points)+'_NN_qualities.pkl')
        
        with open(filename,'wb') as file:
            print("save network")
            pickle.dump(my_net,file)