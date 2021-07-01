# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:26:04 2019

@author: papagian
"""

import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')



from Triangulation import get_extrapoints_target_length_jupyter,get_indices,get_triangle_indices
import Triangulation_with_points
import Triangulation_with_points_old_version
from bilinear_interpolation_grid import *

import pickle


nb_of_edges=14
nb_of_points=6
nb_of_grid_points=20



directory='../polygon_datasets/validation_sets/grid_patch_regression/'
polygon_data=str(nb_of_edges)+'_'+str(nb_of_points)+'_valdation_polygon_data.pkl'
point_coordinates=str(nb_of_edges)+'_'+str(nb_of_points)+'_valdation_point_coordinates.pkl'
estimated_points=str(nb_of_edges)+'_'+str(nb_of_points)+'_'+str(nb_of_grid_points)+'_valdation_estimated_points.pkl'
interpolatd_estimated_points=str(nb_of_edges)+'_'+str(nb_of_points)+'_'+str(nb_of_grid_points)+'_valdation_estimated_interpolated_points.pkl'

polygons=[]
with open(os.path.join(directory,polygon_data),'rb') as f:
    try:
        while True:
            polygons.append(pickle.load(f))
    except EOFError:
        pass


point_coordinates_list=[]
with open(os.path.join(directory,point_coordinates),'rb') as f:
    try:
        while True:
            point_coordinates_list.append(pickle.load(f))
    except EOFError:
        pass
    
with open(os.path.join(directory,estimated_points),'rb') as f:
    estimated_points=pickle.load(f)
    
with open(os.path.join(directory,interpolatd_estimated_points),'rb') as f:
   interpolatd_estimated_points=pickle.load(f)
    
  
case_number=5
contour=polygons[case_number]

point=estimated_points[case_number]
#point=np.array(point_coordinates_list[case_number])

#point=sort_points(point.reshape(1,2*nb_of_points),nb_of_points)
point=point.reshape(nb_of_points,2)
#point=np.array(point).reshape(nb_of_points,2)

contour_with_points=np.vstack([contour,point])
filename='../network_datasets/connectivity_NN/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_NN_qualities_with_extra_points.pkl'
contour_with_point_variable=Variable(torch.from_numpy(contour_with_points.reshape(-1)).type(torch.FloatTensor)).expand(100,len(contour_with_points.reshape(-1)))

with open(filename,'rb') as f:
    my_net=pickle.load(f)
my_net=my_net.cpu().eval()

prediction=my_net(contour_with_point_variable).data[0].numpy()
predicted_qualities=prediction.reshape(nb_of_edges,nb_of_edges+nb_of_points)

plot_contour(contour)
predicted_qualities
predicted_ordered_matrix=Triangulation_with_points.order_quality_matrix(predicted_qualities,contour,contour_with_points,check_for_equal=True)
initial_elements,sub_elements=Triangulation_with_points.triangulate(contour,point,predicted_ordered_matrix,recursive=True)





