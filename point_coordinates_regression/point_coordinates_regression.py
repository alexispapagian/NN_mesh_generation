
# coding: utf-8

# In[58]:
import sys
sys.path.insert(0, '../network_datasets/')


from Triangulation import *

import torch
import torch.optim as optim



from matplotlib import pyplot as plt

import torch.nn as nn

from torch.autograd import Variable
from math import atan2,pow,sqrt,acos,pi
from  Neural_network import *

from torch.autograd.function import Function

#get_ipython().run_line_magic('matplotlib', 'tk')


# In[59]:


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle=np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/pi
    if angle>180:
        angle=angle-360
    return angle




# A fucntion to acquire the 2 points from the braycenter and chosen direction
def get_points(barycenter,direction):
    rotation=np.array([[cos(pi),-sin(pi)],
                         [sin(pi),cos(pi)] ])
    point1=np.array(direction+barycenter)
    direction=np.dot(rotation,direction)
    point2=np.array(direction+barycenter)
    return point1, point2
    


# In[60]:


def get_set_nb_of_points(point_coordinates):
    set_of_numbers=set()
    for index,_ in enumerate(point_coordinates):
        set_of_numbers.add(len(point_coordinates[index][0]))
    return set_of_numbers


def get_indices_nb_of_points(set_of_numbers,number_of_points,point_coordinates):
    indices=[]
    if number_of_points not in set_of_numbers:
        return "No such number of points for sample"
    else:
        for index,_ in enumerate(point_coordinates):
            if len(point_coordinates[index][0])==number_of_points:
                indices.append(index)
        return indices
    
def get_polygons_nb_of_points(indices):
    
    pass

def get_edge_lengths(polygon):
    polygon_edge_lengths=np.empty([polygon.shape[0]])
    for index,_ in enumerate(polygon):
        polygon_edge_lengths[index]=np.linalg.norm(polygon[(index+1)%(polygon.shape[0])]-polygon[index])
    return polygon_edge_lengths
    

def extract_lengths_angles(polygon_set,nb_of_points):
    lengths=[]
    angles=[]
    target_edge_length=[]
    for polygons in polygon_set:
        polygon=np.delete(polygons,2*nb_of_points).reshape(nb_of_points,2)
        lengths.append(get_edge_lengths(polygon))
        angles.append(np.array(np.multiply(pi/180,get_polygon_angles(polygon))))
        target_edge_length.append([polygons[2*nb_of_points]])
    data=np.hstack([lengths,angles,target_edge_length])
    return data,lengths,angles
    
def extract_lengths_angles_in_triangle_form(polygon_set,nb_of_points):
    data,lengths,angles=extract_lengths_angles(polygon_set,nb_of_points)
    data_reformed=np.empty([data.shape[0],3*nb_of_points+1])
    for polygon_index,polygon_lengths in enumerate(lengths.copy()):
        data_reformed[polygon_index][0:3*nb_of_points:3]=polygon_lengths[0:nb_of_points]
        data_reformed[polygon_index][1:3*(nb_of_points-1):3]=polygon_lengths[1:nb_of_points]
        data_reformed[polygon_index][3*nb_of_points-2]=polygon_lengths[0]
        data_reformed[polygon_index][2:3*nb_of_points:3]=angles[polygon_index]
        #including target edge length
        data_reformed[polygon_index][-1]=polygon_set[polygon_index][2*nb_of_points]
        
    return data_reformed

def extract_midpoint_directions(points):
    mid_points=[]
    directions=[]
    for points in points:
        for coordinates in points:
            coordinates_reshape=coordinates.reshape(2,2)
            point_A=np.array(coordinates_reshape[0])
            point_B=np.array(coordinates_reshape[1])
            mid_point=(point_A+point_B)/2
            direction=point_B-mid_point
            if direction[0]<0:
                direction=point_A-mid_point
            mid_points.append(mid_point)
            directions.append(direction)
    mid_points=np.array(mid_points)
    directions=np.array(directions)
        
    mid_points_with_directions=np.hstack([mid_points,directions])
    return mid_points_with_directions,mid_points,directions
    


# In[61]:



class myLossfunction(Function):
    
    @staticmethod
    def forward(self,output,target):
        self.save_for_backward(output,target) 

                        
       # output=output.view(int(output.size()[0]/2),2)

        #target=target.view(int(target.size()[0]/2),2)
        distance=torch.nn.PairwiseDistance()
        result=distance(output,target)

        result=torch.FloatTensor(result)
        #self.save_for_backward(result)


        return  result 
    
    
    @staticmethod
    def backward(self,grad_output1):
        input1,target=self.saved_variables
        
        print(input1)
        #distance=torch.nn.PairwiseDistance()(input1.view(int(input1.size()[0]/2),2),target.view(int(target.size()[0]/2),2))
        distance=torch.nn.PairwiseDistance()(input1,target)

        grad_output1=(input1-target)/distance

        
        return grad_output1,None
    
    
class myOtherLossfunction(Function):
    
    @staticmethod
    def forward(self,output,target)->Variable:
        
        torch_sum=0
        for i in range(0,output.size()[1],2):
            euclidean_distance=(output[:,i]-target[:,i]).pow(2)+(output[:,i+1]-target[:,i+1]).pow(2)
            torch_sum+=euclidean_distance
        return  torch_sum

def my_torch_loss_function(a,b)->Variable:
    torch_sum=0
    for i in range(0,a.size()[1],2):
        euclidean_distance=torch.sqrt((a[:,i]-b[:,i]).pow(2)+(a[:,i+1]-b[:,i+1]).pow(2))
        torch_sum+=euclidean_distance
    return  torch_sum   
    
    

    
    
def my_torch_loss_function2(a,b,target_edge_length)->Variable:
    torch_sum=0
    for i in range(0,a.size()[1],2):
        euclidean_distance=torch.sqrt((a[:,i]-b[:,i]).pow(2)+(a[:,i+1]-b[:,i+1]).pow(2))
        torch_sum+=euclidean_distance
    
    torch_sum=torch.div(torch_sum,target_edge_length)
    return  torch_sum   
    
    

    
    
def my_torch_loss_function3(a,b,target_edge_length,l_param)->Variable:
    torch_sum=0
    torch_point_sum=0
    for i in range(0,a.size()[1],2):
        euclidean_distance=torch.sqrt((a[:,i]-b[:,i]).pow(2)+(a[:,i+1]-b[:,i+1]).pow(2))
        distance_between_points=torch.sqrt((a[:,i]-a[:,i+1]).pow(2))
        target_edge_length_distance=torch.sqrt((distance_between_points-target_edge_length).pow(2))
        torch_sum+=euclidean_distance
        torch_point_sum+=target_edge_length
    
    torch_sum=torch.div(torch_sum,target_edge_length)+l_param*torch_point_sum
    return  torch_sum   
    


# In[62]:


# Using convolution network
# O: output dimension
# I: Input dimensiion
# S: Stride
# P: padding
# w: kernel size
# O=(I-w-2*P)/S+1

# Included batch normalization with dropout layers

nb_of_points_output=1

class Conv_net(nn.Module):
    
    def __init__(self):
        super(Conv_net,self).__init__()
        self.conv1=nn.Conv1d(1,14,kernel_size=14,stride=1)
        self.conv2=nn.Conv1d(14,28,kernel_size=2,stride=1)

        self.bn1=nn.BatchNorm1d(num_features=28*9)
        self.fc1=nn.Linear(28*9,12)
        self.dp_1=nn.Dropout(p=0.5)

        
        self.bn2=nn.BatchNorm1d(num_features=12)
        self.fc2=nn.Linear(12,12)
        self.dp_2=nn.Dropout(p=0.5)

        
        self.bn3=nn.BatchNorm1d(num_features=12)
        self.fc3=nn.Linear(12,12)
        self.dp_3=nn.Dropout(p=0.5)
        
        self.bn4=nn.BatchNorm1d(num_features=12)
        self.fc4=nn.Linear(12,nb_of_points_output*2)
        
    def forward(self,x):
        x=F.relu(F.max_pool1d(self.conv1(x),kernel_size=2,stride=1))
        
        #x=F.relu(F.max_pool1d(self.conv2(x),kernel_size=2,stride=1))
        #x=self.conv2(x)

        x=F.relu(F.max_pool1d(self.conv2(x),kernel_size=2,stride=1))
        x=F.relu(self.fc1(self.bn1(x.view(-1,28*9))))
        x=F.relu(self.fc2( self.dp_1(self.bn2(x))))
        x=F.relu(self.fc3(self.dp_2(self.bn3(x))))

        x=self.fc4(self.bn4(self.dp_3(x)))

        return x


# In[63]:


# Another convolution network involving the pairing of lengths and angles through convolution and then 
# fully connectiong in a  linear fashion
# Input : a matrix X where lengths take X/2 and angles X/2 and the target edge length

class alt_conv_net(nn.Module):
    
    def __init__(self,nb_of_filters,nb_of_hidden_nodes,out_dimension):
        super(alt_conv_net,self).__init__()
        
        self.nb_of_filters=nb_of_filters
        
        self.conv1=nn.Sequential(nn.Conv1d(in_channels=1,out_channels=nb_of_filters,stride=1,kernel_size=2),
                                 nn.MaxPool1d(stride=1,kernel_size=2),nn.ReLU(inplace=True))
        self.conv2=nn.Sequential(nn.Conv1d(in_channels=1,out_channels=nb_of_filters,stride=1,kernel_size=2),
                                 nn.MaxPool1d(stride=1,kernel_size=2),nn.ReLU(inplace=True))
        # The linear connections are fixed for 12 polygon example
   
        self.fc=nn.Sequential(  nn.BatchNorm1d(num_features=nb_of_filters*10+nb_of_filters*10+1),

                                nn.Linear(nb_of_filters*10+nb_of_filters*10+1,nb_of_hidden_nodes),
                                        nn.ReLU(inplace=True),
                               nn.BatchNorm1d(num_features=nb_of_hidden_nodes),
                                  nn.Linear(nb_of_hidden_nodes,nb_of_hidden_nodes),
                                        nn.ReLU(inplace=True),
                               nn.BatchNorm1d(num_features=nb_of_hidden_nodes),
                               nn.Linear(nb_of_hidden_nodes,out_dimension) )                   
    
    
        
        
        
        
    def forward(self,x):
        
        lenghts=x.narrow(1,0,1).narrow(2,0,12)
        angles=x.narrow(1,0,1).narrow(2,12,12)
        target_edge_length=x.narrow(1,0,1).narrow(2,24,1).resize(x.size()[0],1)

       
        conv_result1=self.conv1(lenghts)
        conv_result2=self.conv2(angles)        
        
        # reshape the convolution results
        
        conv_result1=conv_result1.view(-1,self.nb_of_filters*10)
        conv_result2=conv_result2.view(-1,self.nb_of_filters*10)
        
        concat_tensor=torch.cat([conv_result1,conv_result2,target_edge_length],1)
        output=self.fc(concat_tensor)
        return output
        

        
        


# In[64]:


# Another convolutional network. The input of the network is : Length[index], Length[index+1], angle[index] e.g the triangles
# formed by connecting the neighbouring edges of the contour.
# The input is convoluted by a kernel of size 3 with a stride of 3.


class triangle_convoluting_net(nn.Module):
    
    def __init__(self,nb_of_filters,nb_of_hidden_nodes,out_dimension):
        super(triangle_convoluting_net,self).__init__()
        self.nb_of_filters=nb_of_filters
        
        self.conv=nn.Sequential(nn.Conv1d(in_channels=1,out_channels=nb_of_filters,stride=3,kernel_size=3),
                                #nn.Conv1d(in_channels=nb_of_filters,out_channels=nb_of_filters,stride=1,kernel_size=2),
                                nn.MaxPool1d(stride=1,kernel_size=2),nn.ReLU(inplace=True),
                                nn.Conv1d(in_channels=nb_of_filters,out_channels=nb_of_filters,kernel_size=2,stride=1),
                                nn.MaxPool1d(stride=1,kernel_size=2),nn.ReLU(inplace=True))
       
        # The linear connections are fixed for 12 polygon example
   
        self.fc=nn.Sequential(  nn.BatchNorm1d(num_features= nb_of_filters*9+1),
                              
                                # 1rst layer
                                nn.Linear(nb_of_filters*9+1,nb_of_hidden_nodes),
                                        nn.ReLU(inplace=True),
                               nn.BatchNorm1d(num_features=nb_of_hidden_nodes),
                             
                              # 2nd layer
                              nn.Linear(nb_of_hidden_nodes,nb_of_hidden_nodes),
                                 nn.ReLU(inplace=True),
                               nn.BatchNorm1d(num_features=nb_of_hidden_nodes),
                                
                              # 3rd layer
                             nn.Linear(nb_of_hidden_nodes,nb_of_hidden_nodes),
                                  nn.ReLU(inplace=True),
                              nn.BatchNorm1d(num_features=nb_of_hidden_nodes),
                              
                            
                              nn.Linear(nb_of_hidden_nodes,out_dimension)
                             
                             
                             
                             )      

    def forward(self,x):
        
        triangles=x.narrow(1,0,1).narrow(2,0,3*12)
        target_edge_length=x.narrow(1,0,1).narrow(2,3*12,1).resize(x.size()[0],1)
        
        conv_result=self.conv(triangles)
        
        
        
        conv_result=conv_result.view(-1,self.nb_of_filters*9)
        #print(conv_result,target_edge_length)

        concat_tensor=torch.cat([conv_result,target_edge_length],1)
        output=self.fc(concat_tensor)
        
        return output

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

    


# In[8]:


# Plotting the number of polygons with the number of points inserted
def plot_population_nb_of_points(set_of_points,point_coordinates):
    number_of_polygons=[]
    for numbers in set_of_points:
        indixes=get_indices_nb_of_points(set_of_points,numbers,point_coordinates)
        number_of_polygons.append(len(indixes))

    
    #plt.xticks(np.array(list(set_of_points)))
    #plt.yticks(np.array(number_of_polygons))
    plt.figure(figsize=(24, 24))  
    plt.xticks(list(set_of_points))
    plt.yticks(number_of_polygons)
    plt.xlabel('number of insertion points')
    plt.ylabel('Polygon counts')
    plt.title('Plot bar for number of polygons with number of insertion points (20000 total)')
    plt.bar(list(set_of_points), number_of_polygons, color='g')


# In[65]:


def save_network(filename,net):
    path=os.path.join('networks',filename)

    with open(path,'wb') as output:
        pickle.dump(net,output)
        
def load_network(filename):
    path=os.path.join('networks',filename)
        
    with open(path,'rb') as input:
        net=pickle.load(input)
        
    net.eval()
    return net


# In[66]:


# Get dataset of polygons with a specific target edge length 
def get_polygons_target_edge_length(polygons_reshaped,point_coordinates,target_edge_length):
    polygons_reshaped_target_edge_length=polygons_reshaped[polygons_reshaped[:,int(len(polygons_reshaped))]==target_edge_length]

    target_lentgh_indices=np.where(polygons_reshaped[:,int(len(polygons_reshaped))]==target_edge_length)
    point_coordinates_target_edge_length=point_coordinates[target_lentgh_indices]

    point_coordinates=point_coordinates_target_edge_length
    polygons_reshaped=polygons_reshaped_target_edge_length
    
    
# Given a neural network and a test set for this generate more polygons pertubing those that had an error more than tol
def generate_data(net,x_test,tol):
    net=net.cpu()
    x_test=x_test.cpu()
    net.eval()
    predictions=net(x_test).data.numpy()
    target_edge_lengths=x_test[:,16]
    contours=x_test.data.numpy()
    nb_sampling=10
    
    pertubed_contours=[]
    edge_lengths=[]
    inserted_points_delaunay=[]
    set_of_pertubed_contours=set()
    
    
    for index,prediction in enumerate(predictions):
            contour=contours[index]
            contour=np.delete(contour,16)
            contour=contour.reshape(8,2)
            _,real_value=get_extrapoints_target_length(contour,target_edge_lengths[index],algorithm='del2d')
            error=np.linalg.norm(prediction-np.array(real_value))

            if error > tol and hash(tuple(contour.reshape(16))) not in set_of_pertubed_contours:
                print("Pertubing contour, " ,contour)
                set_of_pertubed_contours.add(hash(tuple(contour.reshape(16))))
                print(set_of_pertubed_contours)
                for j in range(8):
                    for i in range(0,nb_of_sampling):
                        pertubed_contour=contour.copy()
                        pertubed_contour[j]=np.random.normal(contour[j],0.1)
                        print(contour[j],"changed to ",pertubed_contour[j])
                        print(pertubed_contour,contour)
                        
                        for edge_length in np.linspace(0.1,1,10):
                            print("Examining target edge length ",edge_length,"for ",pertubed_contour)
                            nb_of_points_delaunay,point_coords_delaunay=get_extrapoints_target_length(pertubed_contour,edge_length,algorithm='del2d')
                            if nb_of_points_delaunay==1:
                                
                                inserted_points_delaunay.append(point_coords_delaunay)
                                print("Inserting pertubed_contour",pertubed_contour)
                                pertubed_contours.append(pertubed_contour)
                                edge_lengths.append(edge_length)  
                                #print("List includes",pertubed_contours)
                                #plot_contour(pertubed_contour)
                                #plt.scatter(point_coords_delaunay[:,0],point_coords_delaunay[:,1])

    return    pertubed_contours,edge_lengths,inserted_points_delaunay


# Given a neural network and a test set for this generate more polygons pertubing those that had an error more than tol
# and for a specific target edge length
def generate_data_for_target_edge_length(net,x_test,tol,target_edge_length):
    net=net.cpu()
    x_test=x_test.cpu()
    net.eval()
    predictions=net(x_test).data.numpy()
    target_edge_lengths=x_test[:,16]
    contours=x_test.data.numpy()
    nb_sampling=10
    
    pertubed_contours=[]
    edge_lengths=[]
    inserted_points_delaunay=[]
    set_of_pertubed_contours=set()
    
    
    for index,prediction in enumerate(predictions):
            contour=contours[index]
            contour=np.delete(contour,16)
            contour=contour.reshape(8,2)
            _,real_value=get_extrapoints_target_length(contour,target_edge_lengths[index],algorithm='del2d')
            error=np.linalg.norm(prediction-np.array(real_value))

            if error > tol and hash(tuple(contour.reshape(16))) not in set_of_pertubed_contours:
                print("Pertubing contour, " ,contour)
                set_of_pertubed_contours.add(hash(tuple(contour.reshape(16))))
                print(set_of_pertubed_contours)
                for j in range(8):
                    for i in range(0,nb_of_sampling):
                        pertubed_contour=contour.copy()
                        pertubed_contour[j]=np.random.normal(contour[j],0.1)
                        print(contour[j],"changed to ",pertubed_contour[j])
                        print(pertubed_contour,contour)
                        
                        print("Examining target edge length ",edge_length,"for ",pertubed_contour)
                        nb_of_points_delaunay,point_coords_delaunay=get_extrapoints_target_length(pertubed_contour,target_edge_length,algorithm='del2d')
                        if nb_of_points_delaunay==1:
                                
                            inserted_points_delaunay.append(point_coords_delaunay)
                            print("Inserting pertubed_contour",pertubed_contour)
                            pertubed_contours.append(pertubed_contour)
                            edge_lengths.append(target_edge_length)  
                                #print("List includes",pertubed_contours)
                                #plot_contour(pertubed_contour)
                                #plt.scatter(point_coords_delaunay[:,0],point_coords_delaunay[:,1])

    return    pertubed_contours,edge_lengths,inserted_points_delaunay


# In[67]:


# Recovering the data of polygons where a specific number of points was inserted
def reshape_data(polygons,coordinates,numb_of_ins_points,nb_of_points):
    polygons=np.array([i for i in polygons for j in range(10)])
    polygons_reshaped=[]
    for polygon in polygons:
        polygons_reshaped.append(polygon.reshape(2,polygons.shape[1]))

    polygons_reshaped=np.array(polygons_reshaped)
    set_of_points=get_set_nb_of_points(coordinates)        
    indices=get_indices_nb_of_points(set_of_points,nb_of_points,coordinates)
    indices=np.asarray(indices)
    number_of_insertion_points=np.array(numb_of_ins_points)
    polygons_reshaped.resize(len(coordinates),2*polygons.shape[1])

    polygons_reshaped=np.hstack([polygons_reshaped[indices],number_of_insertion_points[indices,1].reshape(len(indices),1) ])
    coordinates=[ coordinates[i][0]for i in indices]
    coordinates=np.array(coordinates)
    coordinates=coordinates.reshape(polygons_reshaped.shape[0],1,2*nb_of_points)
    return polygons_reshaped,coordinates


def extract_data(polygons,coordinates,numb_of_ins_points,nb_of_points):
    polygons_reshaped=[]
    for polygon in polygons:
        polygons_reshaped.append(polygon.reshape(2,polygons.shape[1]))

    polygons_reshaped=np.array(polygons_reshaped)
    set_of_points=get_set_nb_of_points(coordinates)     
    #print(set_of_points)
    indices=get_indices_nb_of_points(set_of_points,nb_of_points,coordinates)
    indices=np.asarray(indices)
    number_of_insertion_points=np.array(numb_of_ins_points)
    polygons_reshaped.resize(len(coordinates),2*polygons.shape[1])

    polygons_reshaped=np.hstack([polygons_reshaped[indices],number_of_insertion_points[indices,1].reshape(len(indices),1) ])
    coordinates=[ coordinates[i][0]for i in indices]
    coordinates=np.array(coordinates)
    coordinates=coordinates.reshape(polygons_reshaped.shape[0],1,2*nb_of_points)
    return polygons_reshaped,coordinates





# In[68]:


# Plot a prediction of a random contour based on a trained network (FFN)
def plot_random_prediction_ffn(polygon_edges,net,number_of_points):
    plt.clf()
    for i in range(1000):
        random_contour=apply_procrustes(generate_contour(polygon_edges))
        random_nb_of_points,random_point_coordinated_delaunay=get_extrapoints_target_length(random_contour,0.8,algorithm='del2d')
        if random_nb_of_points==number_of_points:
            break
    random_contour_reshaped=random_contour.reshape(1,2*polygon_edges)
    random_contour_with_target=np.hstack([random_contour_reshaped,[[1]]])



    plot_contour(random_contour)
    random_point_coordinated_delaunay=np.array(random_point_coordinated_delaunay)
    random_point_coordinated_delaunay.reshape(number_of_points,2)
    plt.scatter(random_point_coordinated_delaunay[:,0],random_point_coordinated_delaunay[:,1],label='Point location')




    random_x_variable=Variable(torch.from_numpy(random_contour_with_target))
    random_x_variable=random_x_variable.expand(1000,2*polygon_edges+1).type(torch.FloatTensor)
    net=net.cpu()
    random_prediction=net(random_x_variable)
    random_prediction=random_prediction.data[0].numpy()
    random_prediction=random_prediction.reshape(number_of_points,2)
    plt.scatter(random_prediction[:,0],random_prediction[:,1],label=' fnn prediction')
    plt.legend()
    
    
    
# Plot a prediction of a random contour based on a trained network (CNN)
def plot_random_prediction_cnn(polygon_edges,net,number_of_points):
    plt.clf()
    for i in range(1000):
        random_contour=apply_procrustes(generate_contour(polygon_edges))
        random_nb_of_points,random_point_coordinated_delaunay=get_extrapoints_target_length(random_contour,0.8,algorithm='del2d')
        if random_nb_of_points==number_of_points:
            break
    random_contour_reshaped=random_contour.reshape(1,2*polygon_edges)
    random_contour_with_target=np.hstack([random_contour_reshaped,[[1]]])
    triangles_with_edges=extract_lengths_angles_in_triangle_form(random_contour_with_target,polygon_edges)



    plot_contour(random_contour)
    random_point_coordinated_delaunay=np.array(random_point_coordinated_delaunay)
    random_point_coordinated_delaunay.reshape(number_of_points,2)
    plt.scatter(random_point_coordinated_delaunay[:,0],random_point_coordinated_delaunay[:,1],label='Point location')




    random_x_variable=Variable(torch.from_numpy(triangles_with_edges))
    print(random_x_variable)
    random_x_variable=random_x_variable.expand(1000,1,3*polygon_edges+1).type(torch.FloatTensor)
    net=net.cpu()
    random_prediction=net(random_x_variable)
    random_prediction=random_prediction.data[0].numpy()
    random_prediction=random_prediction.reshape(number_of_points,2)
    plt.scatter(random_prediction[:,0],random_prediction[:,1],label='cnn prediction')
    plt.legend()


# In[146]:


def train_network(net,epochs,optimization,loss_function,variable_x,variable_x_test,variable_y,variable_y_test):
    batch_size=int(variable_x.size()[0]/149 )
    net=net.cuda()

    # Train the network #
    
    net.train()
    for t in range(nb_of_epochs):
        sum_loss=0
        for b in range(0,variable_x.size(0),batch_size):
            out = net(variable_x.narrow(0,b,batch_size)) # input x and predict based on x
            loss=loss_function(out, variable_y.narrow(0,b,batch_size).resize(batch_size,nb_of_points*1)).sum()
            sum_loss+=loss.data[0]       
            optimization.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimization.step()        # apply gradients
            
        if t%10==0: 
            net.eval()
            out_test=net(variable_x_test)  
            test_loss=loss_function(out_test, variable_y_test.resize(len(variable_y_test),2)).sum()
            print("Epoch:",t,"Training Loss:",sum_loss/(variable_x.size(0))," Test loss: ",test_loss.data[0]/(variable_x_test.size(0)))
            net.train()
    
    


# In[70]:



def max_element(A):
    r, (c, l) = max(map(lambda t: (t[0], max(enumerate(t[1]), key=lambda v: v[1])), enumerate(A)), key=lambda v: v[1][1])
    return (l, r, c)


# Precision error for two points
def precision_error(real_points,predictions):
    distances=np.empty([2,2])
    for point_index,point in enumerate(real_points):
        for prediction_index,prediction in enumerate(predictions):
               distances[point_index][prediction_index]=np.linalg.norm(point-prediction)
    
    max_distance,prediction_index,point_index=max_element(distances)
    distance1=distances[prediction_index][(point_index+1)%2]
    distance2=distances[(prediction_index+1)%2][point_index]
    return max(distance2,distance1)


# In[71]:




def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
    return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
    cosx=dot_product(v,w)/(length(v)*length(w))
    rad=acos(cosx) # in radians
    return rad*180/pi # returns degrees
def angle_counterclockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: 
        return 360-inner
    else: 
        return inner

    
    

def extract_barycenter(points,nb_of_points):
        
    barycenters=[]
    
    polygons=points.reshape(len(points),nb_of_points,2)
    for polygon in polygons:
        barycenter=np.array([polygon[:,0].sum()/nb_of_points,polygon[:,1].sum()/nb_of_points])
        barycenter_triangles=[]
        barycenters.append(barycenter)
        for i in range(nb_of_points):
            barycenter_triangles.append(np.array([barycenter,polygon[i],polygon[(i+1)%nb_of_points]]))
    barycenters=np.array(barycenters).reshape(len(points),1,2)
    
    return np.array(barycenters)


def sort_points(point_coordinates,nb_of_points):
    polygon=point_coordinates.reshape(len(point_coordinates),nb_of_points,2)
    barycenters=extract_barycenter(point_coordinates,nb_of_points)
    angles=[]
    polygons=point_coordinates.reshape(len(point_coordinates),nb_of_points,2)
    vectors=polygons-barycenters

    for  barycenter_vectors in vectors:
        for vector in barycenter_vectors:
            angles.append(angle_counterclockwise(np.array([1,0]),vector))
                      
    angles=np.array(angles).reshape(len(vectors),nb_of_points,1)
    point_coordinates_with_angles=np.dstack([polygons,angles])
    point_coordinates_sorted=[]
    for points in point_coordinates_with_angles:
        points_sorted=np.array(sorted(points,key=lambda x: x[2]))
        points_sorted=points_sorted[:,0:2]
        point_coordinates_sorted.append(points_sorted.reshape(1,nb_of_points,2))
    return np.array(point_coordinates_sorted)   


def regression_error(real_points,prediction_points,nb_of_interior_points):
    
    real_point_polygon=sort_points(real_points.reshape(1,1,nb_of_interior_points,2),nb_of_interior_points).reshape(nb_of_interior_points,2)
    prediction_points_polygon=sort_points(prediction_points.reshape(1,1,nb_of_interior_points,2),nb_of_interior_points).reshape(nb_of_interior_points,2)
    procrusted_prediction_points_polygon=apply_procrustes(prediction_points_polygon,real_point_polygon)
    
    error_in_transformation=np.array([np.linalg.norm(prediction_points_polygon[indices]-procrusted_prediction_points_polygon[indices]) for indices in range(len(prediction_points_polygon))])

    maximum_index,maximum_distance=np.argmax([np.linalg.norm(real_point_polygon[index]-procrusted_prediction_points_polygon[index]) for index in range(len(real_point_polygon))]),np.max([np.linalg.norm(real_point_polygon[index]-procrusted_prediction_points_polygon[index]) for index in range(len(real_point_polygon))])
    regression_error=maximum_distance+error_in_transformation[maximum_index]
    
    return regression_error   




