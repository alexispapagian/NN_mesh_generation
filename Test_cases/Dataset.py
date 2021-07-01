# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:28:45 2019

@author: papagian
"""

import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')

sys.path.insert(0, '../network_datasets')
import pickle
import torch
from functools import reduce
from Triangulation import *
from point_coordinates_regression import *
import Triangulation_with_points 



import torch
import torch.optim as optim



#from matplotlib import pyplot as plt

import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
from math import atan2,pow,acos
from  Neural_network import *

from torch.autograd.function import Function

import torch
from torch.utils.data import Dataset,DataLoader

def loadall(filename):
    lis=[]
    with open(filename, "rb") as f:
        while True:
            try:
                lis.append(pickle.load(f,encoding="latin1"))
            except EOFError:
                break
    return lis

def batch_size_factor(n,minimum,maximum):    
    factor_set=set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    for factor in factor_set:
        if factor>minimum and factor<maximum:
            batch_size_factor=factor
    
    return batch_size_factor
        
def reshape_data_for_conv2d(polygons_with_points,qualities):
    polygons_with_points_reshaped=np.empty([len(polygons_with_points),nb_of_edges+nb_of_points+1,2])
    for index,polygon_with_points in enumerate(polygons_with_points):
        polygon_with_points=polygon_with_points.reshape(nb_of_edges+nb_of_points,2)
        polygon_with_points=np.insert(polygon_with_points,nb_of_edges,polygon_with_points[0],axis=0)
        #polygon_with_points=np.vstack([polygon_with_points,polygon_with_points[nb_of_edges+1]])
        polygons_with_points_reshaped[index]=polygon_with_points
    qualities=np.array(qualities)
    qualities_reshaped=qualities.reshape(len(qualities),qualities.shape[1]*qualities.shape[2])
    return polygons_with_points_reshaped,qualities_reshaped

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
        

class PolygonTrainDataset():
    
    def __init__(self,nb_of_edges,nb_of_points):
        polygons_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/qualities_with_sampling/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_grid_sampling'
        quality_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/qualities_with_sampling/min_qualities_with_grid_sampling'
        polygons=loadall(polygons_filepath)
        qualities=loadall(quality_filepath)
        polygons=np.array(polygons)
        qualities=np.array(qualities)
        polygons_with_points_reshaped,qualities_reshaped=reshape_data_for_conv2d(polygons,qualities)
        polygons_with_points_reshaped,qualities_reshaped=unison_shuffled_copies(polygons_with_points_reshaped,qualities_reshaped)

        total_population=len(polygons)
        nb_of_test_data=int(0.2*total_population)
        nb_training_data=total_population-nb_of_test_data
        nb_training_data=total_population

        self.len=nb_training_data

        self.x_data=torch.from_numpy(polygons_with_points_reshaped[:nb_training_data]).type(torch.FloatTensor)
        self.y_data=torch.from_numpy(qualities_reshaped[:nb_training_data]).type(torch.FloatTensor)
        
        
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
        
    def __len__(self):
        return self.len
    

class PolygonTestDataset():
    
    def __init__(self,nb_of_edges,nb_of_points):
        polygons_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/qualities_with_sampling/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_grid_sampling'
        quality_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/qualities_with_sampling/min_qualities_with_grid_sampling'
        polygons=loadall(polygons_filepath)
        qualities=loadall(quality_filepath)
        polygons=np.array(polygons)
        qualities=np.array(qualities)
        polygons_with_points_reshaped,qualities_reshaped=reshape_data_for_conv2d(polygons,qualities)
        polygons_with_points_reshaped,qualities_reshaped=unison_shuffled_copies(polygons_with_points_reshaped,qualities_reshaped)

        total_population=len(polygons)
        nb_of_test_data=int(0.2*total_population)
        nb_training_data=total_population-nb_of_test_data
        self.len=nb_of_test_data
        x_tensor=torch.from_numpy(polygons_reshaped[nb_of_training_data:]).type(torch.FloatTensor)

        y_tensor=torch.from_numpy(qualities_reshaped[nb_of_training_data:]).type(torch.FloatTensor)

        self.x_data=torch.from_numpy(polygons_with_points_reshaped[nb_training_data:]).type(torch.FloatTensor)
        self.y_data=torch.from_numpy(qualities_reshaped[nb_training_data:]).type(torch.FloatTensor)
        
        
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
        
    def __len__(self):
        return self.len
    
if __name__ == '__main__':   
    nb_of_edges=4
    nb_of_points=4
        
    training_dataset=PolygonTrainDataset(nb_of_edges,nb_of_points)
    batch_size_div=batch_size_factor(training_dataset.len,10,600)
    batch_size=int(training_dataset.len/batch_size_div)
    train_loader=DataLoader(dataset=training_dataset,
                            batch_size= batch_size,
                            shuffle=True,
                            num_workers=8)
#    
#    test_dataset=PolygonTestDataset(nb_of_edges,nb_of_points)
#    test_loader=DataLoader(dataset=test_dataset,
#                           batch_size=test_dataset.len,
#                            shuffle=False,
#                            num_workers=4)
#    
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    my_net=alt_2d_conv_net(nb_of_filters=nb_of_edges+2,nb_of_hidden_nodes=(int)(1.5*nb_of_edges*(nb_of_points+nb_of_edges)),out_dimension=training_dataset.y_data.size()[1],nb_of_edges=nb_of_edges,nb_of_points=nb_of_points)
    #torch.cuda.empty_cache()
    print("Training data length:",training_dataset.len)
    print(" Batch size :", batch_size)

    optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-3,weight_decay=0.2)
    loss_func =torch.nn.MSELoss(size_average=False)         
    my_net.to(device)
    loss_func.to(device)
    
    
    for epoch in range(1000):
        training_sum_loss=0
        for i ,data in enumerate(train_loader,0):
            
            #get the inputs
            inputs,labels=data
            
            #wrap the labels
            inputs,labels=Variable(inputs).resize(inputs.size()[0],1,inputs.size()[1],inputs.size()[2]).to(device), Variable(labels).to(device)
            
            #Forward pass: compute
            y_pred=my_net(inputs)
            
            #compute and print loss
            loss=loss_func(y_pred,labels)

            training_sum_loss+=loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch,"Training loss for NN3 ",str(nb_of_edges),"_",str(nb_of_points),": ",training_sum_loss/training_dataset.len)

#        with torch.set_grad_enabled(False):
#            for i ,data in enumerate(test_loader,0):
#                
#                #get the inputs
#                inputs,labels=data
#                
#                #wrap the labels
#                inputs,labels=Variable(inputs,volatile=True).resize(inputs.size()[0],1,inputs.size()[1],inputs.size()[2]).to(device), Variable(labels,volatile=True).to(device)
#                
#                #Forward pass: compute
#                y_test_pred=my_net(inputs)
#                
#                #compute and print loss
#                test_loss=loss_func(y_test_pred,labels).detach().item()
#    
#       print(epoch,"Training loss for NN3 ",str(nb_of_edges),"_",str(nb_of_points),": ",training_sum_loss/training_dataset.len,"Test loss:",test_loss/test_dataset.len)
       
    net_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/networks/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_NN_conn_conv.pkl'
    with open(net_filepath,'wb') as f:
         pickle.dump(my_net,f)