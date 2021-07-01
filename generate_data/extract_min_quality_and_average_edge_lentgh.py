# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 13:23:40 2019

@author: papagian
"""

import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')


import os
from Triangulation import *

edges=[12,13]
for nb_of_edges in edges:
    polygons=load_dataset(str(nb_of_edges)+'_polygons.pkl')
    mean_quality_list=[]
    min_quality_list=[]
    edge_lengths_list=[]
    mean_edge_length_list=[]
    elements_list=[]
    count=0
    for contour in polygons:
        for targe_edge_length in np.linspace(0.1,1,10):
            _,points=get_extrapoints_target_length(contour,targe_edge_length,algorithm='del2d')
            file_directory='..\\gmsh\\'
            filename=os.path.join(file_directory,'output.msh')
            if len(points)!=0:
                contour_with_points=np.vstack([contour,points])
            else:
                contour_with_points=contour
        
    
            triangles_indices=get_triangle_indices(contour)
            elements_list.append(triangles_indices)
            element_edge_length_list=[]
            edges_mean_length=[]
            for triangle_indices in triangles_indices:
                triangle=contour_with_points[np.asarray(triangle_indices)]
                print(triangle)
                triangle_edge_lengths=[]
                for i in range(3):
                    triangle_edge_lengths.append(compute_edge_lengths(triangle[int(i)],triangle[int((i+1)%3)]))
                    element_edge_length_list.append(triangle_edge_lengths)
                triangle_edge_lengths_mean=np.array(triangle_edge_lengths).mean()
                edges_mean_length.append(triangle_edge_lengths_mean)
            edge_lengths_list.append(element_edge_length_list)
            edges_mean_length=np.array(edges_mean_length)
            min_quality,mean_quality=compute_minimum_quality_triangulated_contour(contour_with_points,triangles_indices),compute_mean_quality_triangulated_contour(contour_with_points,triangles_indices)
            print("Number of elements is:",len(triangles_indices))
            print("Minimum quality is : ", min_quality)
            print("Mean quality is : ", mean_quality)
            print("Mean edge length is:",edges_mean_length.mean())
            #print(" edge lengths are:",edge_lengths_list)
            mean_edge_length_list.append(edges_mean_length.mean())
            min_quality_list.append(min_quality)
            mean_quality_list.append(mean_quality)
            
            
            
            
            home_directory='..//polygon_datasets//'
            home_filename1=os.path.join(home_directory,str(nb_of_edges)+'_mesh_mean_quality.pkl')
            home_filename2=os.path.join(home_directory,str(nb_of_edges)+'_mesh_min_quality.pkl')
            home_filename3=os.path.join(home_directory,str(nb_of_edges)+'_edge_length_mean.pkl')
            home_filename4=os.path.join(home_directory,str(nb_of_edges)+'edge_lengths.pkl')
            home_filename5=os.path.join(home_directory,str(nb_of_edges)+'_elements.pkl')
            count+=1
            print("Wrinting ",count ,"out of :", 10* len(polygons))

            with open(home_filename1,'ab') as f:
                pickle.dump([mean_quality],f)
            with open(home_filename2,'ab') as f:
                pickle.dump([min_quality],f)
            with open(home_filename3,'ab')as f:
                pickle.dump([edges_mean_length.mean()],f)
            with open(home_filename5,'ab')as f:
                pickle.dump([triangles_indices],f)
            with open(home_filename4,'ab')as f:
                pickle.dump([element_edge_length_list],f)
                
        
        
    
       
    


            
        