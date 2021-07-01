# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 19:20:17 2020

@author: papagian
"""

# generate polygon datasets including a required number of insertion Delaunay points

import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')
import os

from Triangulation import *
from point_coordinates_regression import *


number_of_samples=50
edges=[3]
required_inner_points=1

for nb_of_edges in edges: 
    polygon_list=[]
    point_coordinates_list=[]
    target_edge_length_list=[]
    count=0
    stop=False
    polygons_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'
    grid_qualities_path=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/patch_scores/'

    target_edge_length=1
    while  not stop:
         polygon=apply_procrustes(generate_contour(nb_of_edges))
         nb_of_points,point_coordinates=get_extrapoints_target_length_additional(polygon,target_edge_length=1,algorithm='del2d')
         if nb_of_points==required_inner_points:
             count+=1
             print(count," out of ",str(number_of_samples),"for "+str(nb_of_edges)+" polygons")

             polygon_with_point=np.vstack([polygon,point_coordinates]).reshape(-1)
             polygon_with_point_target_length=np.hstack([polygon.reshape(-1),target_edge_length])

             if not os.path.exists(polygons_filepath):
                 os.makedirs(polygons_filepath)
            
             if not os.path.exists(grid_qualities_path):
                 os.makedirs(grid_qualities_path)
                 
             with open(os.path.join(polygons_filepath,str(nb_of_edges)+'_'+str(required_inner_points)+'_polygons.pkl'),'ab') as f:
                 pickle.dump(polygon_with_point,f)
                   
             with open(os.path.join(grid_qualities_path,str(nb_of_edges)+'_'+str(required_inner_points)+'_polygons_with_target_length.pkl'),'ab') as f:
                 pickle.dump(polygon_with_point_target_length,f)
             
             
         if count==number_of_samples:
                 stop=True