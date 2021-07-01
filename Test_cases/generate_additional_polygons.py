# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:14:50 2019

@author: papagian
"""

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

number_of_samples=100
edges=[10]
required_inner_points=10

for nb_of_edges in edges: 
    polygon_list=[]
    point_coordinates_list=[]
    target_edge_length_list=[]
    count=0
    stop=False
    
    polygons_with_points=[]
    while  not stop:
        polygon=apply_procrustes(generate_contour(nb_of_edges))
        for target_edge_length in (0.3,1,10):
             nb_of_points,point_coordinates=get_extrapoints_target_length_additional(polygon,target_edge_length,algorithm='del2d')
             if nb_of_points==required_inner_points:
                 point_coordinates=np.array(point_coordinates)
                 print(count," out of ",str(number_of_samples),"for "+str(nb_of_edges)+" polygons")
                 polygon_with_points=np.hstack([polygon.reshape(1,-1),point_coordinates.reshape(1,-1)])
                 homedirectory='additional_datasets/'
                 with open(os.path.join(homedirectory,str(nb_of_edges)+'_'+str(required_inner_points)+'_polygons_part2.pkl'),'ab') as f:
                         pickle.dump(polygon_with_points,f)
                 count+=1   
                 break       
        if count==number_of_samples:
            stop=True