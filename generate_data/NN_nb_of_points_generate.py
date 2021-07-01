# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 09:50:59 2019

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


def batch_size_factor(n,minimum,maximum):    
    factor_set=set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    for factor in factor_set:
        if factor>minimum and factor<maximum:
            batch_size_factor=factor
    return batch_size_factor

edges=[14,15]
for nb_of_edges in edges:
    filedirectory='..//polygon_datasets'
    filename_polygons=os.path.join(filedirectory,str(nb_of_edges)+'_polygons.pkl')
    filename_points=os.path.join(filedirectory,str(nb_of_edges)+'_nb_of_points_del.pkl')
    
    with open(filename_polygons,'rb') as file:
        polygons=pickle.load(file)
        
    with open(filename_points,'rb') as file:
        nb_of_points=pickle.load(file)
        
    polygons=[i for i in polygons for j in range(10)]
    
    polygons_reshaped=[]
    for polygon in polygons:
        polygons_reshaped.append(polygon.reshape(1,2*nb_of_edges))
        
    for index,polygon in enumerate(polygons_reshaped):
        if (index+1)%10==0:
            polygons_reshaped[index]=np.append(polygon,1)
        else:    
            polygons_reshaped[index]=np.append(polygon,((0.1*((index+1)%10))))
            
    nb_of_points=[i[0] for i in nb_of_points]
    nb_of_points=np.array(nb_of_points)
    polygons_reshaped=np.array(polygons_reshaped)
    
    polygons_reshaped,nb_of_points=unison_shuffled_copies(polygons_reshaped,nb_of_points)
    
    
    nb_of_test_data=int(len(polygons_reshaped)*0.2)
    nb_of_training_data=int(len(polygons_reshaped)-nb_of_test_data)
    
    
    x_tensor=torch.from_numpy(polygons_reshaped[:nb_of_training_data]).type(torch.FloatTensor)
    x_tensor_test=torch.from_numpy(polygons_reshaped[nb_of_training_data:]).type(torch.FloatTensor)
    x_variable,x_variable_test=Variable(x_tensor),Variable(x_tensor_test)
    
    y_tensor=torch.from_numpy(nb_of_points[:nb_of_training_data]).type(torch.FloatTensor)
    y_tensor_test=torch.from_numpy(nb_of_points[nb_of_training_data:]).type(torch.FloatTensor)
    y_variable,y_variable_test=Variable(y_tensor),Variable(y_tensor_test)
    
    my_net=Net(2*nb_of_edges+1,1,nb_of_hidden_layers=2, nb_of_hidden_nodes=12*nb_of_edges,batch_normalization=True)
    torch.cuda.empty_cache()
    
    
    optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-4,weight_decay=1e-4)
    loss_func = torch.nn.MSELoss(size_average=False) 
    
    
   if  torch.cuda.is_available():
        my_net.cuda()
        loss_func.cuda()
        x_variable , y_variable,x_variable_test,y_variable_test= x_variable.cuda(), y_variable.cuda(),x_variable_test.cuda(),y_variable_test.cuda()
        print("cuda activated")
    
   
    training_data_size=int(x_variable.size()[0])
    print("Training data size: ",training_data_size)
    batch_size_div=batch_size_factor(training_data_size,500,1000)
    batch_size=int(training_data_size/batch_size_div)
    nb_of_epochs=1000
    my_net.cuda()
    #my_net.cpu()
    # Train the network #
    for t in range(nb_of_epochs):
        sum_loss=0
        for b in range(0,x_variable.size(0),batch_size):
            out = my_net(x_variable.narrow(0,b,batch_size))                 # input x and predict based on x
            loss = loss_func(out, y_variable.narrow(0,b,batch_size))     # must be (1. nn output, 2. target), the target label is NOT one-hotted
            sum_loss+=loss.data[0]
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            #print(t,loss.data[0])
            optimizer.step()        # apply gradients
        if t%10==0:
            my_net.eval()
            test_loss=loss_func(my_net(x_variable_test),y_variable_test).data[0]
            my_net.train()
            print("Epoch:",t,"Training Loss:",sum_loss/x_variable.size(0),"Epoch:",t,"Test Loss:",test_loss/x_variable_test.size(0))
    
    my_net=my_net.eval()
    home_directory='..//network_datasets//nb_of_points_NN'
    filename_network=os.path.join(home_directory,str(nb_of_edges)+'_nn_nb_of_points.pkl')
    
    with open(filename_network,'wb') as file:
        pickle.dump(my_net,file)