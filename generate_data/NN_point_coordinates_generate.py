# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:14:40 2019

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
from functools import reduce


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
    
def batch_size_factor(n,minimum,maximum):    
    factor_set=set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    for factor in factor_set:
        if factor>minimum and factor<maximum:
            batch_size_factor=factor
    
    return batch_size_factor
    
edges=[5,6,7,8]
for nb_of_edges in edges:
    polygons=load_dataset(str(nb_of_edges)+'_polygons.pkl')
    del_points=load_dataset(str(nb_of_edges)+'_point_coordinates_del.pkl')
    number_of_insertion_points=load_dataset(str(nb_of_edges)+'_nb_of_points_del.pkl')
    
    
    set_points=get_set_nb_of_points(del_points)
    
    
    for  nb_of_inner_points in set_points:
        if nb_of_inner_points==0:
            continue
        
        nb_of_grid_points=10
        
        X=np.linspace(-1.3,1.3,nb_of_grid_points)
        Y=np.linspace(-1.3,1.3,nb_of_grid_points)
        XX,YY=np.meshgrid(X,Y)
        
        grid_points=np.array([[x,y] for x in X for y in Y])
        
        
        
        
        polygons_reshaped,point_coordinates=reshape_data(polygons,del_points,number_of_insertion_points,nb_of_inner_points)
        
        
        
        # Including the grid points to the polygon set
        
        polygons_reshaped_with_grid_points=[]
        for polygon in polygons_reshaped:
            for index in range(len(grid_points)):
                polygons_reshaped_with_grid_points.append( np.hstack([polygon,grid_points[index]]))
        polygons_reshaped_with_grid_points=np.array(polygons_reshaped_with_grid_points)
        
        
        
        # For every polygon obtain grid qualities
        
        polygons_grid_qualities=[]
        for index,points in enumerate(point_coordinates):
            inner_points=points.reshape(nb_of_inner_points,2)
            contour=np.delete(polygons_reshaped[index],2*nb_of_edges).reshape(nb_of_edges,2)
            #grid_qualities=get_grid_qualities_with_penalty(XX,YY,inner_points,contour)
            #grid_qualities=get_grid_qualities(XX,YY,inner_points)
            grid_qualities=get_grid_qualities_with_penalty_midpoint_included(XX,YY,inner_points,contour)
        
            polygons_grid_qualities.append(grid_qualities)
            
        
        
        
        polygons_grid_qualities=np.array(polygons_grid_qualities)
        polygons_grid_qualities_reshaped=polygons_grid_qualities.reshape(len(polygons_reshaped)*len(grid_points),1,1)
        
        # Shuffle the data
        polygons_reshaped_with_grid_points,polygons_grid_qualities_reshaped=unison_shuffled_copies(polygons_reshaped_with_grid_points,polygons_grid_qualities_reshaped)
        
        
        
        
        # 80/20 training/test data ratio
        
        nb_of_test_data=int(len(polygons_reshaped_with_grid_points)*0.2)
        nb_of_training_data=int(len(polygons_reshaped_with_grid_points)-nb_of_test_data)
        nb_of_test_data,nb_of_training_data
        
        
        
        # Setting up the variables
        
        x_tensor=torch.from_numpy(polygons_reshaped_with_grid_points[:nb_of_training_data]).type(torch.FloatTensor)
        x_tensor_test=torch.from_numpy(polygons_reshaped_with_grid_points[nb_of_training_data:]).type(torch.FloatTensor)
        x_variable,x_variable_test=Variable(x_tensor),Variable(x_tensor_test)
        
        y_tensor=torch.from_numpy(polygons_grid_qualities_reshaped[:nb_of_training_data]).type(torch.FloatTensor)
        y_tensor_test=torch.from_numpy(polygons_grid_qualities_reshaped[nb_of_training_data:]).type(torch.FloatTensor)
        
        y_variable,y_variable_test=Variable(y_tensor),Variable(y_tensor_test)
        
        
        my_net=Net(x_variable.size()[1],y_variable.size()[2],nb_of_hidden_layers=2, nb_of_hidden_nodes=40,batch_normalization=True)
        torch.cuda.empty_cache()
        print("Training data length:",x_variable_test.size()[1],y_variable.size()[2])
        
        
        optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-4,weight_decay=0.2)
        loss_func =torch.nn.MSELoss(size_average=False) 
        
        
        if  torch.cuda.is_available():
            loss_func.cuda()
                
            x_variable , y_variable=x_variable.cuda(), y_variable.cuda()
            x_variable_test,y_variable_test= Variable(x_tensor_test.cuda(),volatile=True),Variable(y_tensor_test.cuda(),volatile=True)
        
            print("cuda activated")
            
            
        training_data_size=int(x_variable.size()[0])
        print("Training data size: ",training_data_size)
        batch_size_div=batch_size_factor(training_data_size,20,2600)
        batch_size=int(training_data_size/batch_size_div)
        nb_of_epochs=1 
        my_net.cuda()
        #my_net.cpu()
        
        # Train the network #
        my_net.train()
        for t in range(nb_of_epochs):
            sum_loss=0
            for b in range(0,x_variable.size(0),batch_size):
                out = my_net(x_variable.narrow(0,b,batch_size))                 # input x and predict based on x
                loss= loss_func(out, y_variable.narrow(0,b,batch_size))     # must be (1. nn output, 2. target), the target label is NOT one-hotted
                
                sum_loss+=float(loss.data[0])
        
                
                optimizer.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
                del loss
                del out
            if t%10==0: 
                my_net.eval()
                out_test=my_net(x_variable_test)   
                test_loss=loss_func(out_test,y_variable_test)
                print("Epoch:",t,"Training Loss:",sum_loss/(x_variable.size(0)),test_loss.data[0]/(x_variable_test.size(0)))
                my_net.train()
                
                
        home_directory='..//network_datasets//grid_NN'
        filename=os.path.join(home_directory,str(nb_of_edges)+'_'+str(nb_of_inner_points)+'_'+str(nb_of_grid_points)+'_grid_regression_NN.pkl')
                
        with open(filename,'wb') as file:
            pickle.dump(my_net,file)