# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:59:36 2019

@author: papagian
"""

# generate polygon datasets including a required number of insertion Delaunay points

import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')

from Triangulation import *
from point_coordinates_regression import *

number_of_samples=1000
edges=[14]
required_inner_points=11

for nb_of_edges in edges: 
    polygon_list=[]
    point_coordinates_list=[]
    target_edge_length_list=[]
    count=0
    stop=False
    
    while  not stop:
        polygon=apply_procrustes(generate_contour(nb_of_edges))
        for target_edge_length in (0.3,1,10):
             nb_of_points,point_coordinates=get_extrapoints_target_length_additional(polygon,target_edge_length,algorithm='del2d')
             if nb_of_points==required_inner_points:
                 polygon_list.append(polygon)
                 point_coordinates_list.append(point_coordinates)
                 target_edge_length_list.append(target_edge_length)
                 count+=1
                 print(count," out of ",str(number_of_samples),"for "+str(nb_of_edges)+" polygons")
                 homedirectory='../polygon_datasets/validation_sets/grid_patch_regression'
                 if count==1:                     
                     with open(os.path.join(homedirectory,str(nb_of_edges)+'_'+str(required_inner_points)+'_valdation_polygon_data.pkl'),'wb') as f:
                         pickle.dump(polygon,f)
                     with open(os.path.join(homedirectory,str(nb_of_edges)+'_'+str(required_inner_points)+'_valdation_point_coordinates.pkl'),'wb') as f:
                         pickle.dump([point_coordinates],f)
                     with open(os.path.join(homedirectory,str(nb_of_edges)+'_'+str(required_inner_points)+'_valdation_target_edge_lengths.pkl'),'wb') as f:
                         pickle.dump(target_edge_length,f)
                 else:
                     with open(os.path.join(homedirectory,str(nb_of_edges)+'_'+str(required_inner_points)+'_valdation_polygon_data.pkl'),'ab') as f:
                         pickle.dump(polygon,f)
                     with open(os.path.join(homedirectory,str(nb_of_edges)+'_'+str(required_inner_points)+'_valdation_point_coordinates.pkl'),'ab') as f:
                         pickle.dump([point_coordinates],f)
                     with open(os.path.join(homedirectory,str(nb_of_edges)+'_'+str(required_inner_points)+'_valdation_target_edge_lengths.pkl'),'ab') as f:
                         pickle.dump(target_edge_length,f)
                        
                    
                    
                 break
        if count==number_of_samples:
            stop=True
                 
                