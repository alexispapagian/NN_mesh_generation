# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:11:46 2019

@author: papagian
"""


import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../network_datasets')
sys.path.insert(0,'../point_coordinates_regression')




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


def get_median_edge_length_population(nb_of_edges,nb_of_points):
    with open('../polygon_datasets/'+str(nb_of_edges)+'_polygons_with_points/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_points','rb') as file:
        polygons_with_points=pickle.load(file)

    polygons_edges_lengths=[]
    for polygon_with_points in polygons_with_points:
        triangles=get_elements(polygon_with_points[:2*nb_of_edges].reshape(nb_of_edges,2),polygon_with_points[2*nb_of_edges:].reshape(nb_of_points,2))
        triangles=np.array(triangles)
        triangles=triangles.reshape(len(triangles),3,2)

        edge_lengths_list=[]
        element_edge_length_list=[]
        for triangle in triangles:
            triangle_edge_lengths=[]
            for i in range(3):
                triangle_edge_lengths.append(compute_edge_lengths(triangle[int(i)],triangle[int((i+1)%3)]))
            element_edge_length_list.append(triangle_edge_lengths)
            triangle_edge_lengths_mean=np.array(triangle_edge_lengths).mean()
        edge_lengths_list.append(element_edge_length_list)
        polygons_edges_lengths.append(edge_lengths_list)
    merged=list(itertools.chain(*polygons_edges_lengths))
    merged = list(itertools.chain(*merged))
    merged = list(itertools.chain(*merged))
    return np.median(merged)


def get_elements(polygon,points):
    
    
    triangles_in_mesh=[]
    contour_connectivity=get_contour_edges(polygon)
    polygon_with_points=np.vstack([polygon,points])
    shape=dict(vertices=polygon_with_points,segments=contour_connectivity)

    t = tri.triangulate(shape, 'pq0')

    for triangle_index in t['triangles']:
        triangles_in_mesh.append(polygon_with_points[np.asarray([triangle_index])])
    return triangles_in_mesh


def unique_permutations(iterable, r=None):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p



def l2_relative_error(A,B):
    A=np.array(A)
    B=np.array(B)
    diff=abs(B-A).flatten()
    maximum_error_index=np.argmax(diff)
    return np.linalg.norm(B-A)/np.linalg.norm(A),maximum_error_index


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
    

def seperate_to_sectors(grid_points,nb_sectors):
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


def get_qualities_by_sector(grid_points,inner_points,contour,nb_sectors,nb_of_edges,nb_of_points,outing_zone):
    
    

    
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
                        if np.linalg.norm(grid_point-contour_point)<outing_zone:
                            quality=1.3
                            

                        one_third=0.25*contour[index]+0.75*contour[(index+1)%len(contour)]
                        mid_point=0.5*(contour[index]+contour[(index+1)%len(contour)])
                        two_thirds=0.75*contour[index]+0.25* contour[(index+1)%len(contour)]
                        if np.linalg.norm(grid_point-mid_point)<outing_zone or np.linalg.norm(grid_point-one_third)<outing_zone or np.linalg.norm(grid_point-two_thirds)<outing_zone:
                            quality=1.3
                    quality_point.append(quality)
            quality_sectors.append(quality_point) 
    return np.array(quality_sectors)
    

    
    


def get_qualities_by_sector_updated(grid_points,inner_points,contour,nb_sectors,nb_of_edges,nb_of_points,outing_zone):
      
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
                        if np.linalg.norm(grid_point-contour_point)<outing_zone:
                            quality=1.3

                        one_third=0.25*contour[index]+0.75*contour[(index+1)%len(contour)]
                        two_thirds=0.75*contour[index]+0.25* contour[(index+1)%len(contour)]


                        mid_point=0.5*(contour[index]+contour[(index+1)%len(contour)])

                        one_fith=0.2*contour[index]+0.8*contour[(index+1)%len(contour)]
                        two_fifth=(2/5)*contour[index]+(3/5)*contour[(index+1)%len(contour)]
                        third_fifth=(3/5)*contour[index]+(2/5)*contour[(index+1)%len(contour)]
                        fourth_fith=0.8*contour[index]+0.2*contour[(index+1)%len(contour)]

                        one_sixth=(1/6)*contour[index]+(5/6)*contour[(index+1)%len(contour)]
                        two_sixth=(2/6)*contour[index]+(6/6)*contour[(index+1)%len(contour)]
                        fourth_sixth=(4/6)*contour[index]+(2/6)*contour[(index+1)%len(contour)]
                        fifth_sixth=(5/6)*contour[index]+(1/6)*contour[(index+1)%len(contour)]
                        condition1=np.linalg.norm(grid_point-one_sixth)<outing_zone or  np.linalg.norm(grid_point-two_sixth)<outing_zone or  np.linalg.norm(grid_point-fourth_sixth)<outing_zone or  np.linalg.norm(grid_point-fifth_sixth)<outing_zone
                        condition2=np.linalg.norm(grid_point-one_fith)<outing_zone or  np.linalg.norm(grid_point-two_fifth)<outing_zone or np.linalg.norm(grid_point-third_fifth)<outing_zone or  np.linalg.norm(grid_point-fourth_fith)<outing_zone or  np.linalg.norm(grid_point-fifth_sixth)<outing_zone

                        if  condition2 or condition1 or np.linalg.norm(grid_point-mid_point)<outing_zone or np.linalg.norm(grid_point-one_third)<outing_zone or np.linalg.norm(grid_point-two_thirds)<outing_zone:
                            quality=1.3
                    quality_point.append(quality)
            quality_sectors.append(quality_point) 
    return np.array(quality_sectors)
    
    
    

def get_qualities_by_sector_original(grid_points,inner_points,contour,nb_sectors):
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




class PolygonTrainDataset():
    
    def __init__(self,nb_of_edges,nb_of_points):
        polygon_target_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/patch_scores/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_target_length.pkl'
        grid_score_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/patch_scores/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_20_sector_qualities.pkl'
#        with open(polygon_target_filepath,'rb') as f :
#            polygons_with_target_edge_length=pickle.load(f)
#        with open(grid_score_filepath,'rb') as f:
#            polygons_qualities_sector=pickle.load(f)  
#           
        polygons_with_target_edge_length=loadall(polygon_target_filepath)

        polygons_with_target_edge_length=np.array(polygons_with_target_edge_length).reshape(len(polygons_with_target_edge_length),2*nb_of_edges+1)
        polygons_qualities_sector=loadall(grid_score_filepath)[0]
        polygons_qualities_sector=np.array(polygons_qualities_sector)

        polygons_reshaped_with_sector_grid_points=[]

        for polygon in polygons_with_target_edge_length:
            for sector in sectors:
                polygons_reshaped_with_sector_grid_points.append( np.hstack([polygon,sector.reshape(2*len(sector))]))
            
        polygons_reshaped_with_sector_grid_points=np.array(polygons_reshaped_with_sector_grid_points)
        
        # Shuffle the data
        polygons_qualities_sector=polygons_qualities_sector.reshape(int(polygons_qualities_sector.shape[0]*polygons_qualities_sector.shape[1]),1
                                                                ,int(polygons_qualities_sector.shape[2]))
        polygons_reshaped_with_sector_grid_points,polygons_qualities_sector=unison_shuffled_copies(polygons_reshaped_with_sector_grid_points,polygons_qualities_sector)
        
        total_population=len(polygons_reshaped_with_sector_grid_points)
        nb_of_test_data=int(0.2*total_population)
        nb_training_data=total_population-nb_of_test_data
        self.len=nb_of_test_data
        
        self.x_data=torch.from_numpy(polygons_reshaped_with_sector_grid_points[:nb_training_data]).type(torch.FloatTensor)
        self.y_data=torch.from_numpy(polygons_qualities_sector[:nb_training_data]).type(torch.FloatTensor)
        
          
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
        
    def __len__(self):
        return self.len

class PolygonTestDataset():
    
    def __init__(self,nb_of_edges,nb_of_points):
        polygon_target_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/patch_scores/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_target_length.pkl'
        grid_score_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/patch_scores/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_20_sector_qualities.pkl'
#        with open(polygon_target_filepath,'rb') as f :
#            polygons_with_target_edge_length=pickle.load(f)
#        with open(grid_score_filepath,'rb') as f:
#            polygons_qualities_sector=pickle.load(f)  
#           
        polygons_with_target_edge_length=loadall(polygon_target_filepath)
        polygons_with_target_edge_length=np.array(polygons_with_target_edge_length).reshape(len(polygons_with_target_edge_length),2*nb_of_edges+1)
        polygons_qualities_sector=loadall(grid_score_filepath)[0]
        polygons_qualities_sector=np.array(polygons_qualities_sector)
        polygons_reshaped_with_sector_grid_points=[]

        for polygon in polygons_with_target_edge_length:
            for sector in sectors:
                polygons_reshaped_with_sector_grid_points.append( np.hstack([polygon,sector.reshape(2*len(sector))]))
            
        polygons_reshaped_with_sector_grid_points=np.array(polygons_reshaped_with_sector_grid_points)
        
        # Shuffle the data
        polygons_qualities_sector=polygons_qualities_sector.reshape(int(polygons_qualities_sector.shape[0]*polygons_qualities_sector.shape[1]),1
                                                                ,int(polygons_qualities_sector.shape[2]))
        polygons_reshaped_with_sector_grid_points,polygons_qualities_sector=unison_shuffled_copies(polygons_reshaped_with_sector_grid_points,polygons_qualities_sector)
        
        total_population=len(polygons_reshaped_with_sector_grid_points)
        nb_of_test_data=int(0.2*total_population)
        nb_training_data=total_population-nb_of_test_data
        self.len=nb_of_test_data
        
        self.x_data=torch.from_numpy(polygons_reshaped_with_sector_grid_points[nb_training_data:]).type(torch.FloatTensor)
        self.y_data=torch.from_numpy(polygons_qualities_sector[nb_training_data:]).type(torch.FloatTensor)
        
        
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
        
    def __len__(self):
        return self.len

def loadall(filename):
    lis=[]
    with open(filename, "rb") as f:
        while True:
            try:
                lis.append(pickle.load(f,encoding="latin1"))
            except EOFError:
                break
    return lis
    
if __name__ == '__main__':   
    nb_of_edges=5
    nb_of_points=1
        

    nb_of_grid_points=20
    X=np.linspace(-1.3,1.3,nb_of_grid_points)
    Y=np.linspace(-1.3,1.3,nb_of_grid_points)
    XX,YY=np.meshgrid(X,Y)
    
    grid_points=np.array([[x,y] for x in X for y in Y])
    
    nb_sectors=int(nb_of_grid_points/2)
    sectors,indices=seperate_to_sectors(grid_points,nb_sectors)
    
      
    training_dataset=PolygonTrainDataset(nb_of_edges,nb_of_points)
    batch_size_div=batch_size_factor(training_dataset.len,10,60)
    batch_size=int(training_dataset.len/batch_size_div)
    train_loader=DataLoader(dataset=training_dataset,
                            batch_size= batch_size,
                            shuffle=True,
                            num_workers=0)
    
    test_dataset=PolygonTestDataset(nb_of_edges,nb_of_points)
    test_loader=DataLoader(dataset=test_dataset,
                           batch_size=test_dataset.len,
                            shuffle=False,
                            num_workers=0)
    
     
    use_cuda = torch.cuda.is_available()
#   ! device = torch.device("cuda:0" if use_cuda else "cpu")
    device = torch.device("cpu")


    
    my_net=Net(training_dataset.x_data.size()[1],training_dataset.y_data.size()[2],nb_of_hidden_layers=4, nb_of_hidden_nodes=280,batch_normalization=True)
    
    
    #torch.cuda.empty_cache()
    print("Training data length:",training_dataset.len)
    print(" Batch size :", batch_size)

    optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-4,weight_decay=0.2)
    loss_func =torch.nn.MSELoss(size_average=False)         
    my_net.to(device)
    loss_func.to(device)
    
    
    for epoch in range(1000):
        training_sum_loss=0
        for i ,data in enumerate(train_loader,0):
            
            #get the inputs
            inputs,labels=data
            
            #wrap the labels
            inputs,labels=Variable(inputs).to(device), Variable(labels).reshape(labels.size()[0],labels.size()[2]).to(device)
            
            #Forward pass: compute
            y_pred=my_net(inputs)
            
            #compute and print loss
            loss=loss_func(y_pred,labels)
            training_sum_loss+=loss.item()

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#        with torch.set_grad_enabled(False):
#            for i ,data in enumerate(test_loader,0):
#                
#                #get the inputs
#                inputs,labels=data
#                
#                #wrap the labels
#                inputs,labels=Variable(inputs).to(device), Variable(labels).reshape(labels.size()[0],labels.size()[2]).to(device)
#                
#                #Forward pass: compute
#                y_test_pred=my_net(inputs)
#                
#                #compute and print loss
#                test_loss=loss_func(y_test_pred,labels).item()
        print(epoch,"Training NN2 loss for: ",str(nb_of_edges),"_",str(nb_of_points)," ",training_sum_loss/training_dataset.len)

#        print(epoch,"Training NN2 loss for: ",str(nb_of_edges),"_",str(nb_of_points)," ",training_sum_loss/training_dataset.len,"Test loss:",test_loss/test_dataset.len)
    network_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/networks/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_'+str(nb_of_grid_points)+'_grid_NN'
    with open(network_filepath,'wb') as f:
        pickle.dump(my_net,f)