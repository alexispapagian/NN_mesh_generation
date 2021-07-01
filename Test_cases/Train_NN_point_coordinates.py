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




get_ipython().run_line_magic('matplotlib', 'qt')


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

# Assemble data


nb_of_edges=12

nb_of_points=11


nb_of_grid_points=20
X=np.linspace(-1.3,1.3,nb_of_grid_points)
Y=np.linspace(-1.3,1.3,nb_of_grid_points)
XX,YY=np.meshgrid(X,Y)

grid_points=np.array([[x,y] for x in X for y in Y])


nb_sectors=int(nb_of_grid_points/2)
sectors,indices=seperate_to_sectors(grid_points,nb_sectors)

polygons_qualities_sector=load_grid_sector_qualities(str(nb_of_edges)+'_'+str(nb_of_points)+'_'+str(nb_of_grid_points)+'_sector_qualities_additional_v4.pkl')


polygons_initial=load_dataset(str(nb_of_edges)+'_polygons.pkl')
number_of_insertion_points=load_dataset(str(nb_of_edges)+'_nb_of_points_del.pkl')
point_coordinates_initial=load_dataset(str(nb_of_edges)+'_point_coordinates_del.pkl')








polygons_reshaped,point_coordinates=reshape_data(polygons_initial,point_coordinates_initial,number_of_insertion_points,int(nb_of_points))



additional_polygons=[]
with open(os.path.join('../polygon_datasets/additional_polygon_datasets/'+str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl'),'rb') as f:
        try:
            while True:
                additional_polygons.append(pickle.load(f))
        except EOFError:
            pass


additional_target_edge_lengths=[]
with open(os.path.join('../polygon_datasets/additional_polygon_datasets/'+str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_target_edge_lengths.pkl'),'rb') as f:
        try:
            while True:
                additional_target_edge_lengths.append(pickle.load(f))
        except EOFError:
            pass

additional_point_coordinates=[]        
with open(os.path.join('../polygon_datasets/additional_polygon_datasets/'+str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_point_coordinates.pkl'),'rb') as f:
        try:
            while True:
                additional_point_coordinates.append(pickle.load(f))
        except EOFError:
            pass







additional_polygons=np.array(additional_polygons[-1])
additional_polygons=additional_polygons.reshape(len(additional_polygons),2*nb_of_edges)
additional_target_edge_lengths=np.array(additional_target_edge_lengths[-1])        
additional_target_edge_lengths=np.array(additional_target_edge_lengths).reshape(len(additional_target_edge_lengths),1)
additional_polygons_reshaped=np.hstack([additional_polygons,additional_target_edge_lengths])
additional_point_coordinates=np.array(additional_point_coordinates[-1]).reshape(len(additional_point_coordinates[-1]),1,2*nb_of_points)


polygons_reshaped=np.vstack([polygons_reshaped,additional_polygons_reshaped])
point_coordinates=np.vstack([point_coordinates,additional_point_coordinates])








polygons_qualities_sector=polygons_qualities_sector.reshape(int(polygons_qualities_sector.shape[0]*polygons_qualities_sector.shape[1]),1
                                                            ,int(polygons_qualities_sector.shape[2]))


# In[204]:


polygons_reshaped_with_sector_grid_points=[]
for polygon in polygons_reshaped:
    for sector in sectors:
        polygons_reshaped_with_sector_grid_points.append( np.hstack([polygon,sector.reshape(2*len(sector))]))
polygons_reshaped_with_sector_grid_points=np.array(polygons_reshaped_with_sector_grid_points)


# In[205]:


# 80/20 training/test data ratio

nb_of_test_data=int(len(polygons_reshaped_with_sector_grid_points)*0.2)
nb_of_training_data=int(len(polygons_reshaped_with_sector_grid_points)-nb_of_test_data)
nb_of_test_data,nb_of_training_data


# In[206]:


# Shuffle the data
polygons_reshaped_with_sector_grid_points,polygons_qualities_sector=unison_shuffled_copies(polygons_reshaped_with_sector_grid_points,polygons_qualities_sector)


# In[207]:


# Setting up the variables

x_tensor=torch.from_numpy(polygons_reshaped_with_sector_grid_points[:nb_of_training_data]).type(torch.FloatTensor)
x_tensor_test=torch.from_numpy(polygons_reshaped_with_sector_grid_points[nb_of_training_data:]).type(torch.FloatTensor)
x_variable,x_variable_test=Variable(x_tensor),Variable(x_tensor_test)

y_tensor=torch.from_numpy(polygons_qualities_sector[:nb_of_training_data]).type(torch.FloatTensor)
y_tensor_test=torch.from_numpy(polygons_qualities_sector[nb_of_training_data:]).type(torch.FloatTensor)

y_variable,y_variable_test=Variable(y_tensor),Variable(y_tensor_test)


# In[208]:


my_net=Net(x_variable.size()[1],y_variable.size()[2],nb_of_hidden_layers=3, nb_of_hidden_nodes=200,batch_normalization=True)
torch.cuda.empty_cache()
print("Training data length:",x_variable_test.size()[1],y_variable.size()[2])
x_variable.size()


# In[209]:


optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-4,weight_decay=0.1)
loss_func =torch.nn.MSELoss(size_average=False) 


# In[210]:



if  torch.cuda.is_available():
    loss_func.cuda()
        
    x_variable , y_variable=x_variable.cuda(), y_variable.cuda()
    x_variable_test,y_variable_test= Variable(x_tensor_test.cuda(),volatile=True),Variable(y_tensor_test.cuda(),volatile=True)

    print("cuda activated")
#    


# In[213]:


training_data_size=int(x_variable.size()[0])
print("Training data size: ",training_data_size)
batch_size_div=batch_size_factor(training_data_size,1000,5000)
batch_size=int(training_data_size/batch_size_div)
print("Batch size: ", batch_size)
nb_of_epochs=3000
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


# In[212]:


save_grid_patch_NN(str(nb_of_edges)+'_'+str(nb_of_points)+'_'+str(nb_of_grid_points)+'grid_NN_additional_v4.pkl',my_net)










