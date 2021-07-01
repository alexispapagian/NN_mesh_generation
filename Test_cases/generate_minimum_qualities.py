
import sys
sys.path.insert(0, '../Triangulation/')

import numpy as np
import pickle
from Triangulation_with_points import *




nb_of_edges=4
nb_of_points=1


try:
    
    with open(str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl','rb') as f:
                polygons=pickle.load(f)
                
                
    with open(str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_point_coordinates.pkl','rb') as f:
                points=pickle.load(f)
    
    common_index=min(len(points),len(polygons))
    
    polygons=polygons[:common_index]
    points=points[:common_index]
    
    polygons=np.array(polygons).reshape(len(polygons),2*nb_of_edges)
    points=np.array(points).reshape(len(polygons),2*nb_of_points)
    polygons_with_points=np.hstack([polygons,points])
    
except:
    polygons=[]
    with open(str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl','rb') as f:
        try:
            while True:
                polygons.append(pickle.load(f))
        except EOFError:
                pass
    
    points=[]
    with  open(str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_point_coordinates.pkl','rb') as f:
        try:
            while True:
                points.append(pickle.load(f))
        except EOFError:
                pass
            
    
        
        polygons=np.array(polygons).reshape(len(polygons),2*nb_of_edges)
        points=np.array(points).reshape(len(polygons),2*nb_of_points)
        polygons_with_points=np.hstack([polygons,points])
    
    



for polygon_with_points in polygons_with_points:
    directory_path=str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/qualities'
    polygon_with_points=polygon_with_points.reshape(int(len(polygon_with_points)/2),2)
    polygon=polygon_with_points[:nb_of_edges]
    inner_points=polygon_with_points[nb_of_edges:]
            
    minimum_quality,mean_quality=quality_matrices(polygon,inner_points)
  

    filename_min=os.path.join(directory_path,str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_qualities_min.pkl')
    filename_mean=os.path.join(directory_path,str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_qualities_mean.pkl')
    with open(filename_min,'ab') as h1:
          pickle.dump(minimum_quality,h1)
    
    with open(filename_mean,'ab') as h2:
           pickle.dump(mean_quality,h2)    