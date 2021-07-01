
# coding: utf-8

# In[15]:
import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')

from Triangulation_with_points import *
from point_coordinates_regression import *


# In[22]:

nb_of_edges=[4]

# In[ ]:

# Dataset for number of points inserted and their point coordinates
for nb in nb_of_edges:
    polygons=load_dataset(str(nb)+'_polygons.pkl')
    nb_of_points=[]
    points_coodinates=[]
    count=0
    for polygon in polygons:
        for i in np.linspace(.1,1,10):
            nb_point,point_coord=get_extrapoints_target_length(polygon,i,algorithm='del2d')
            nb_of_points.append([nb_point,i])
            points_coodinates.append([point_coord,i])
            count+=1
    save_dataset(str(nb)+'_nb_of_points_del.pkl',nb_of_points)
    save_dataset(str(nb)+'_point_coordinates_del.pkl',points_coodinates)


# In[19]:




# In[ ]:



