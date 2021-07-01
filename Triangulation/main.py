# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:16:51 2018

@author: papagian
"""
import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')

from Triangulation import get_extrapoints_target_length,get_indices,get_triangle_indices
from Triangulation_with_points import *
import pickle

#from Triangulation import *

for i in range(49):
    contour=apply_procrustes(generate_contour(6))
    nb_of_points,point=get_extrapoints_target_length(contour,target_edge_length=1,algorithm='del2d')
    if nb_of_points>=1:
        break


#with open('non_working_case4_9.pkl','rb') as handle:
#     contour_with_points=pickle.load(handle)
 
    

#contour=contour_with_points[:9]  
  
#point=contour_with_points[9:]    

    
point=np.array(point)
#point=sort_points(point.reshape(1,len(point),2),len(point)).reshape(len(point),2)

contour_with_points=np.vstack([contour,point])
quality,_=quality_matrix(contour,point)
ordered_matrix=order_quality_matrix(quality,contour,contour_with_points,check_for_equal=True)
triangulate(contour,point,ordered_matrix,recursive=True)
#list_of_elements=concat_element_list(elements,sub_element_list)
#minimum_quality=compute_minimum_quality_triangulated_contour(contour,list_of_elements)
#minimum_quality_del=compute_delaunay_minimum_quality(contour)


plot_contour(contour)
#print(minimum_quality,minimum_quality_del)