# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:23:18 2018

@author: papagian
"""
import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')

def find_unique_indices(a):
    
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _,idx=np.unique(b,return_index=True)
    
    return idx


import os
from Triangulation_with_points import *
from point_coordinates_regression import *
polygons=[16]

for nb_of_edges in polygons:
    directory_home='..//polygon_datasets//'
    directory_name=str(nb_of_edges)+'_polygons_with_points'

    directory_path=os.path.join(directory_home,directory_name)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    
    
    
#    polygons=load_dataset(str(nb_of_edges)+'_polygons.pkl')
#    del_points=load_dataset(str(nb_of_edges)+'_point_coordinates_del.pkl')
#    number_of_insertion_points=load_dataset(str(nb_of_edges)+'_nb_of_points_del.pkl')
#    set_points=get_set_nb_of_points(del_points)
        
    polygons=[]
    with open('../polygon_datasets/'+str(nb_of_edges)+'_polygons.pkl','rb') as f:
        try:
            while True:
                polygons.append(pickle.load(f))
        except EOFError:
                pass
    polygons=np.array(polygons)
    
    del_points=[]
    with open('../polygon_datasets/'+str(nb_of_edges)+'_point_coordinates_del.pkl','rb') as f:
        try:
            while True:
                del_points.append(pickle.load(f))
        except EOFError:
                pass
 #   del_points=np.array(del_points)
            
            
    number_of_insertion_points=[]
    with open('../polygon_datasets/'+str(nb_of_edges)+'_nb_of_points_del.pkl','rb') as f:
        try:
            while True:
                number_of_insertion_points.append(pickle.load(f))
        except EOFError:
                pass
    number_of_insertion_points=np.array(number_of_insertion_points)
#
    target_edge_lengths=[]
    for i in range(len(polygons)*9):
         target_edge_lengths.append((0.2+(i%9)*0.1))
    target_edge_lengths=np.array(target_edge_lengths).reshape(len(target_edge_lengths),1)
    number_of_insertion_points=np.hstack([number_of_insertion_points,target_edge_lengths])
    
    set_points=get_set_nb_of_points(del_points)

    
    for number_of_points in set_points:
            polygons_reshaped,point_coordinates=reshape_data(polygons,del_points,number_of_insertion_points,number_of_points)

            polygons_reshaped=np.delete(polygons_reshaped,int(2*nb_of_edges),1)
            unique_indices=np.sort(find_unique_indices(polygons_reshaped))
            polygons_reshaped_unique=polygons_reshaped[unique_indices]
            point_coordinates_unique=point_coordinates[unique_indices]
            polygons_with_inner_points=np.hstack([polygons_reshaped_unique,point_coordinates_unique.reshape(point_coordinates_unique.shape[0],point_coordinates_unique.shape[2])])


            filepath=os.path.join(directory_path,str(nb_of_edges)+'_'+str(number_of_points)+'_polygons_with_points')
            with open(filepath,'wb') as f:
                pickle.dump(polygons_with_inner_points,f)   