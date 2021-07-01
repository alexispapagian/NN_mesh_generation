# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 15:39:09 2018

@author: papagian
"""

import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')

import os
import pickle
from Triangulation_with_points import *
from point_coordinates_regression import load_dataset,get_set_nb_of_points

edges=[14]
for nb_of_edges in edges:
    directory_home='..//polygon_datasets'
    sub_directory_path=str(nb_of_edges)+'_polygons_with_points'
    directory_path=os.path.join(directory_home,sub_directory_path)
    del_points=load_dataset(str(nb_of_edges)+'_point_coordinates_del.pkl')
    number_of_insertion_points=load_dataset(str(nb_of_edges)+'_nb_of_points_del.pkl')
    set_points=get_set_nb_of_points(del_points)
    
    for nb_of_points in set_points:
        filepath=os.path.join(directory_path,str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_points')
        if nb_of_points==0:
            continue

        with open(filepath,'rb') as f:
            polygons_with_points=pickle.load(f)

        mean_qualities=[]
        min_qualities=[]
        for polygon_with_points in polygons_with_points:
            polygon_with_points=polygon_with_points.reshape(int(len(polygon_with_points)/2),2)
            polygon=polygon_with_points[:nb_of_edges]
            inner_points=polygon_with_points[nb_of_edges:]
            print(" Before sorting: ",inner_points.reshape(int(nb_of_points),2))
            inner_points=sort_points(inner_points.reshape(1,2*nb_of_points),nb_of_points).reshape(nb_of_points,2)
            print(" After sorting: ",inner_points.reshape(int(nb_of_points),2))

            minimum_quality,mean_quality=quality_matrices(polygon,inner_points)
            min_qualities.append(minimum_quality)
            mean_qualities.append(mean_quality)

        filename_min=os.path.join(directory_path,str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_qualities_min.pkl')
        filename_mean=os.path.join(directory_path,str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_qualities_mean.pkl')
        with open(filename_min,'wb') as h1:
            pickle.dump(min_qualities,h1)
    
        with open(filename_mean,'wb') as h2:
            pickle.dump(mean_qualities,h2)    