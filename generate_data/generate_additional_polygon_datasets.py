# generate polygon datasets including a required number of insertion Delaunay points

import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')

from Triangulation import *
from point_coordinates_regression import *

number_of_samples=500
edges=[14,12,10,8]
required_inner_points=1

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
                 homedirectory='../polygon_datasets/additional_polygon_datasets/'
                 directory_name=str(nb_of_edges)+'_polygons'
                 directory_path=os.path.join(homedirectory,directory_name)
                 if not os.path.exists(directory_path):
                     os.makedirs(directory_path)
                 with open(os.path.join(directory_path,str(nb_of_edges)+'_'+str(required_inner_points)+'_polygons.pkl'),'ab') as f:
                     pickle.dump(polygon_list,f)
                 with open(os.path.join(directory_path,str(nb_of_edges)+'_'+str(required_inner_points)+'_point_coordinates.pkl'),'ab') as f:
                     pickle.dump(point_coordinates_list,f)
                 with open(os.path.join(directory_path,str(nb_of_edges)+'_'+str(required_inner_points)+'_target_edge_lengths.pkl'),'wb') as f:
                     pickle.dump(target_edge_length_list,f)
        
                 break
        if count==number_of_samples:
            stop=True
                 
                
   
        
    

    

 
