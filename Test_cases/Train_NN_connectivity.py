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


# Convolution network over the coodinates in  a (nb_of_edges+nb_of_points,2) matrix the first coordinate of the polygon 
# is duplicated at the end of the polygon coordiantes for adjacency information. Same goes for the 1st inner point.




class Dataset(data.Dataset):
    
    'Characterizes a dataset for Pytorch'
    
    def __init__(self,list_IDs,labels):
        ' Initialization'
        self.labels=labels
        self.list_IDs=list_IDs
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self,index):
        ' Generates one sample of data'
        #Select sample
        ID=self.list_IDs[index]
        
        # Load data and get label
        X=torch.load('data/'+ID+'.pt')
        
        
        



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
        conv_result2=self.conv2(inner_points)        
        
        # reshape the convolution results
        
        conv_result1=conv_result1.view(-1,self.nb_of_filters*(2*(self.nb_of_edges-1)))
        conv_result2=conv_result2.view(-1,self.nb_of_filters*(2*(self.nb_of_points-1)))
    
        concat_tensor=torch.cat([conv_result1,inner_points],1)
        output=self.fc(concat_tensor)
        return output
        

        
        
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



    
def batch_size_factor(n,minimum,maximum):    
    factor_set=set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    for factor in factor_set:
        if factor>minimum and factor<maximum:
            batch_size_factor=factor
    
    return batch_size_factor


def loadall(filename):
    lis=[]
    with open(filename, "rb") as f:
        while True:
            try:
                lis.append(pickle.load(f))
            except EOFError:
                break
    return lis




nb_of_edges=14
nb_of_points=6

polygons_filename=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/qualities_with_sampling/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_grid_sampling'
polygons=loadall(polygons_filename)

qualities_filename=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/qualities_with_sampling/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_grid_sampling'
qualities=loadall(qualities_filename)


polygons_with_points_reshaped,qualities_reshaped=reshape_data_for_conv2d(polygons,qualities)

polygons_with_points_reshaped,qualities_reshaped=unison_shuffled_copies(polygons_with_points_reshaped,qualities_reshaped)

nb_of_test_data=int(len(polygons_with_points_reshaped)*0.2)
nb_of_training_data=int(len(polygons_with_points_reshaped)-nb_of_test_data)


x_tensor=torch.from_numpy(polygons_with_points_reshaped[:nb_of_training_data]).type(torch.FloatTensor)
x_tensor_test=torch.from_numpy(polygons_with_points_reshaped[nb_of_training_data:]).type(torch.FloatTensor)
x_variable,x_variable_test=Variable(x_tensor).resize(x_tensor.size()[0],1,x_tensor.size()[1],x_tensor.size()[2]),Variable(x_tensor_test).resize(x_tensor_test.size()[0],1,x_tensor_test.size()[1],x_tensor_test.size()[2])


y_tensor=torch.from_numpy(qualities_reshaped[:nb_of_training_data]).type(torch.FloatTensor)
y_tensor_test=torch.from_numpy(qualities_reshaped[nb_of_training_data:]).type(torch.FloatTensor)
y_variable,y_variable_test=Variable(y_tensor),Variable(y_tensor_test)

my_net=alt_2d_conv_net(nb_of_filters=nb_of_edges+2,nb_of_hidden_nodes=(int)(1.5*nb_of_edges*(nb_of_points+nb_of_edges)),out_dimension=y_variable.size()[1],nb_of_edges=nb_of_edges,nb_of_points=nb_of_points)
#torch.cuda.empty_cache()
print("Training data length:",x_variable.size()[0],y_variable.size()[0])

optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-4,weight_decay=0.2)
loss_func =torch.nn.MSELoss(size_average=False) 

training_data_size=int(x_variable.size()[0])
print("Training data size: ",training_data_size)
batch_size_div=batch_size_factor(training_data_size,1,10000)
batch_size=int(training_data_size/1)
nb_of_epochs=13000 
my_net=my_net.cpu()
#my_net=my_net.cuda()

# Train the network #

my_net.train()

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
    print("Epoch:",t,"Training Loss:",sum_loss/x_variable.size(0),"Test Loss:",test_loss/x_variable_test.size(0))
