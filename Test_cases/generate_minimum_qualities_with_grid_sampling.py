
import sys
sys.path.insert(0, '../Triangulation/')

import numpy as np
import pickle
from Triangulation_with_points import *


import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../point_coordinates_regression/')

sys.path.insert(0, '../network_datasets')
import pickle
import torch


from Triangulation import *
import Triangulation_with_points

from point_coordinates_regression import *

import torch
import torch.optim as optim

import itertools

#from matplotlib import pyplot as plt

import torch.nn as nn

from torch.autograd import Variable
from math import atan2,pow,acos
from  Neural_network import *

from torch.autograd.function import Function
import matplotlib.path as mpltPath
get_ipython().run_line_magic('matplotlib', 'qt')



import matplotlib.cm as cm
import matplotlib.colors as colors
from functools import reduce
import random


def loadall(filename):
    lis=[]
    with open(filename, "rb") as f:
        while True:
            try:
                lis.append(pickle.load(f))
            except EOFError:
                break
    return lis

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0



def ray_tracing(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def get_grid_qualities(XX,YY,inner_points):
    qualities=np.empty([len(XX[0,:]),len(YY[:,0])])
    for index_x,x in enumerate(XX[0,:]):
        for index_y,y in enumerate(YY[:,0]):
            quality=np.min(np.array([np.linalg.norm(point-np.array([x,y]),2) for point in inner_points]))
            qualities[index_x][index_y]=quality
    return qualities

def get_grid_qualities_with_penalty(XX,YY,inner_points,contour):
    qualities=np.empty([len(XX[0,:]),len(YY[:,0])])
    for index_x,x in enumerate(XX[0,:]):
        for index_y,y in enumerate(YY[:,0]):
            quality=np.min(np.array([np.linalg.norm(point-np.array([x,y]),2) for point in inner_points]))
            for contour_point in contour:
                if np.linalg.norm(np.array([x,y])-contour_point)<0.2:
                    quality=2
                
            qualities[index_x][index_y]=quality
    return qualities


def get_grid_qualities_with_penalty_midpoint_included(XX,YY,inner_points,contour):
    qualities=np.empty([len(XX[0,:]),len(YY[:,0])])
    for index_x,x in enumerate(XX[0,:]):
        for index_y,y in enumerate(YY[:,0]):
            quality=np.min(np.array([np.linalg.norm(point-np.array([x,y]),2) for point in inner_points]))
            for index,contour_point in enumerate(contour):
                if np.linalg.norm(np.array([x,y])-contour_point)<0.13:
                    quality=100.3
                mid_point=0.5*(contour[index]+contour[(index+1)%len(contour)])
                if np.linalg.norm(np.array([x,y])-mid_point)<0.13:
                    quality=100.3


                
            qualities[index_x][index_y]=quality
    return qualities


def plot_grid_qualities(contour,grid_qualities,grid_points,inner_points):
    plt.clf()
    B=list(grid_qualities.flatten())
    cs = plt.scatter(grid_points[:,0],grid_points[:,1],c=B,cmap=cm.RdYlGn_r,vmin=min(grid_qualities.flatten()),vmax=max(grid_qualities.flatten()),s=4)
    plot_contour(contour)
    plot_contour(contour)
    plt.scatter(inner_points[:,0],inner_points[:,1],marker='o',c='b',label='Point location')
    plt.colorbar(cs)
    plt.legend()
    plt.show()
    

def seperate_to_sectors(grid_points,nb_sectors):
    grid_step_size=int(nb_of_grid_points/nb_sectors)
    sectors=[] 
    indices=[]
    for q in range(nb_sectors):
        for k in range(nb_sectors):
            sector_points=[]
            for j in range(grid_step_size):
                for i in range(grid_step_size):
                    index=(grid_step_size*k+i+grid_step_size*nb_sectors*j)+(grid_step_size**2)*nb_sectors*q
                    sector_points.append(grid_points[index])
                    indices.append(index)
            sectors.append(sector_points) 
    return np.array(sectors),np.array(indices)


def get_qualities_by_sector(grid_points,inner_points,contour,nb_sectors):
    grid_step_size=int(nb_of_grid_points/nb_sectors)
    quality_sectors=[] 
    for q in range(nb_sectors):
        for k in range(nb_sectors):
            quality_point=[]
            for j in range(grid_step_size):
                for i in range(grid_step_size):
                    index=(grid_step_size*k+i+grid_step_size*nb_sectors*j)+(grid_step_size**2)*nb_sectors*q
                    grid_point=grid_points[index]
                    quality=np.min(np.array([np.linalg.norm(point-grid_point,2) for point in inner_points]))
                    for index,contour_point in enumerate(contour):
                        if np.linalg.norm(grid_point-contour_point)<0.13:
                            quality=10.3
                        mid_point=0.5*(contour[index]+contour[(index+1)%len(contour)])
                        if np.linalg.norm(grid_point-mid_point)<0.13:
                            quality=10.3
                    quality_point.append(quality)
            quality_sectors.append(quality_point) 
    return np.array(quality_sectors)
    
    

def plot_grid_qualities(contour,grid_qualities,grid_points,inner_points):
    B=list(grid_qualities.flatten())
    cs = plt.scatter(grid_points[:,0],grid_points[:,1],c=B,cmap=cm.RdYlGn_r,vmin=min(grid_qualities.flatten()),vmax=max(grid_qualities.flatten()),s=4)
    plot_contour(contour)
    plot_contour(contour)
    plt.scatter(inner_points[:,0],inner_points[:,1],marker='o',c='b',label='Point location')
    plt.colorbar(cs)
  #  plt.legend()
   # plt.show()
    
    
def batch_size_factor(n,minimum,maximum):    
    factor_set=set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
    for factor in factor_set:
        if factor>minimum and factor<maximum:
            div_factor=factor
        
    return div_factor



def ring_neighborhood(index,nb_of_grid_points,region=1):
    if region==1:
        
        one_ring=np.array([index+1,index-1,index+nb_of_grid_points
                        ,index-nb_of_grid_points,index-nb_of_grid_points+1,index-nb_of_grid_points-1,
                        index+nb_of_grid_points+1,index+nb_of_grid_points-1])
        return one_ring
    if region==2:
        two_ring=np.array([index+2,index-2,
                        index+2*nb_of_grid_points,
                        index+2*nb_of_grid_points+2,index+2*nb_of_grid_points-2,
                        index+2*nb_of_grid_points+1,index+2*nb_of_grid_points-1,                      
                                              
                                              
                        index-2*nb_of_grid_points,
                        index-2*nb_of_grid_points+1,index-2*nb_of_grid_points-1, 
                        index-2*nb_of_grid_points+2, index-2*nb_of_grid_points-2 ,
                                              
                        index+2,
                        index-2 ,
                        index+nb_of_grid_points+2,
                        index+nb_of_grid_points-2,
                                              
                        index-nb_of_grid_points+2,
                        index-nb_of_grid_points-2,                                                              
                ])
        return two_ring

def get_median_edge_length_population(nb_of_edges,nb_of_points):
    with open('../polygon_datasets/'+str(nb_of_edges)+'_polygons_with_points/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_points','rb') as file:
        polygons_with_points=pickle.load(file)

    polygons_edges_lengths=[]
    for polygon_with_points in polygons_with_points:
        triangles=get_elements(polygon_with_points[:2*nb_of_edges].reshape(nb_of_edges,2),polygon_with_points[2*nb_of_edges:].reshape(nb_of_points,2))
        triangles=np.array(triangles)
        triangles=triangles.reshape(len(triangles),3,2)

        edge_lengths_list=[]
        element_edge_length_list=[]
        for triangle in triangles:
            triangle_edge_lengths=[]
            for i in range(3):
                triangle_edge_lengths.append(compute_edge_lengths(triangle[int(i)],triangle[int((i+1)%3)]))
            element_edge_length_list.append(triangle_edge_lengths)
            triangle_edge_lengths_mean=np.array(triangle_edge_lengths).mean()
        edge_lengths_list.append(element_edge_length_list)
        polygons_edges_lengths.append(edge_lengths_list)
    merged=list(itertools.chain(*polygons_edges_lengths))
    merged = list(itertools.chain(*merged))
    merged = list(itertools.chain(*merged))
    return np.median(merged)

def get_elements(polygon,points):
    
    
    triangles_in_mesh=[]
    contour_connectivity=get_contour_edges(polygon)
    polygon_with_points=np.vstack([polygon,points])
    shape=dict(vertices=polygon_with_points,segments=contour_connectivity)

    t = tri.triangulate(shape, 'pq0')

    for triangle_index in t['triangles']:
        triangles_in_mesh.append(polygon_with_points[np.asarray([triangle_index])])
    return triangles_in_mesh


def unique_permutations(iterable, r=None):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p




nb_of_edges=4
nb_of_points=4
homedirectory=str(nb_of_edges)+'_polygons/'
try:
#    
#    with open(homedirectory+'/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl','rb') as f:
#                polygons=pickle.load(f)
#    
    polygons_with_points=[]                         
    #with open('additional_datasets/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_part2.pkl','rb') as f:
    with open(homedirectory+'/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl','rb') as f:
        try:            
            while True:
                polygons_with_points.append(pickle.load(f))
        except EOFError:
            pass

##with open('additional_datasets/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl','rb') as f:
##    try:            
##        while True:
##            polygons_with_points.append(pickle.load(f))
##    except EOFError:
##        pass
#
#    
#polygons_with_points=polygons_with_points
    polygons_with_points=np.array(polygons_with_points)[0]
#polygons_with_points=np.array(polygons_with_points).reshape(len(polygons_with_points),2*(nb_of_edges+nb_of_points))
##    common_index=min(len(points),len(polygons))
##    
##    polygons=polygons[:common_index]
##    points=points[:common_index]
#    
##    polygons=np.array(polygons).reshape(len(polygons),2*nb_of_edges)
##    points=np.array(points).reshape(len(polygons[2*nb_of_edges:]),2*nb_of_points)
##    polygons_with_points=np.hstack([polygons,points])
    polygons=polygons_with_points[:,:2*nb_of_edges]
    points=polygons_with_points[:,2*nb_of_edges:]
    polygons_with_points=np.hstack([polygons,points])
#       
#      
#  
#    
#    
except:
    polygons=[]
    with open(homedirectory+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl','rb') as f:
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
    
    







#polygons_with_points=polygons_with_points[0]
nb_of_grid_points=20
    
X=np.linspace(-1.2,1.2,nb_of_grid_points)
Y=np.linspace(-1.2,1.2,nb_of_grid_points)
XX,YY=np.meshgrid(X,Y)

grid_points=np.array([[x,y] for x in X for y in Y])


nb_sectors=int(nb_of_grid_points/2)
sectors,indices=seperate_to_sectors(grid_points,nb_sectors)
    
  
grid_step_size=int(nb_of_grid_points/nb_sectors)
    
sectors,indices=seperate_to_sectors(grid_points,nb_sectors)
    
target_edge_length=get_median_edge_length_population(nb_of_edges,nb_of_points)


for polygon in polygons:
    polygon=polygon.reshape(nb_of_edges,2)
            
    # Population of grid points inside the polygon
    
    count=0
    interior_point_indices=[]
    for index,point in enumerate(grid_points):
        is_inside=ray_tracing(point[0],point[1],polygon)
        if is_inside:
            interior_point_indices.append(index)
            count+=1
    
    
    
    # Exclude points that are near the edges of the polygons
    
    interior_grid_points=grid_points[np.asarray(interior_point_indices)]
    
    forbidden_interior_indices=[]
    
    outing_zone=0.2*target_edge_length
    
    for grid_point_index in (interior_point_indices):
         for index,contour_point in enumerate(polygon):
                if np.linalg.norm(grid_points[grid_point_index]-contour_point)<0.1:
                    forbidden_interior_indices.append(grid_point_index)
                one_third=0.25*polygon[index]+0.75*polygon[(index+1)%len(polygon)]
                two_thirds=0.75*polygon[index]+0.25* polygon[(index+1)%len(polygon)]
    
    
                mid_point=0.5*(polygon[index]+polygon[(index+1)%len(polygon)])
                
                one_fith=0.2*polygon[index]+0.8*polygon[(index+1)%len(polygon)]
                two_fifth=(2/5)*polygon[index]+(3/5)*polygon[(index+1)%len(polygon)]
                third_fifth=(3/5)*polygon[index]+(2/5)*polygon[(index+1)%len(polygon)]
                fourth_fith=0.8*polygon[index]+0.2*polygon[(index+1)%len(polygon)]
                
                one_sixth=(1/6)*polygon[index]+(5/6)*polygon[(index+1)%len(polygon)]
                two_sixth=(2/6)*polygon[index]+(6/6)*polygon[(index+1)%len(polygon)]
                fourth_sixth=(4/6)*polygon[index]+(2/6)*polygon[(index+1)%len(polygon)]
                fifth_sixth=(5/6)*polygon[index]+(1/6)*polygon[(index+1)%len(polygon)]
    
                condition1=np.linalg.norm(grid_points[grid_point_index]-one_sixth)<outing_zone or  np.linalg.norm(grid_points[grid_point_index]-two_sixth)<outing_zone or  np.linalg.norm(grid_points[grid_point_index]-fourth_sixth)<outing_zone or  np.linalg.norm(grid_points[grid_point_index]-fifth_sixth)<outing_zone
                condition2=np.linalg.norm(grid_points[grid_point_index]-one_fith)<outing_zone or  np.linalg.norm(grid_points[grid_point_index]-two_fifth)<outing_zone or np.linalg.norm(grid_points[grid_point_index]-third_fifth)<outing_zone or  np.linalg.norm(grid_points[grid_point_index]-fourth_fith)<outing_zone or  np.linalg.norm(grid_points[grid_point_index]-fifth_sixth)<outing_zone
                if condition1 or condition2 or np.linalg.norm(grid_points[grid_point_index]-fourth_fith)<outing_zone or np.linalg.norm(grid_points[grid_point_index]-one_fith)<outing_zone or np.linalg.norm(grid_points[grid_point_index]-mid_point)<outing_zone or np.linalg.norm(grid_points[grid_point_index]-one_third)<outing_zone or np.linalg.norm(grid_points[grid_point_index]-two_thirds)<outing_zone:
                    forbidden_interior_indices.append(grid_point_index)
    #Population of grid points to be sampled under restrictions
    
    set_allowed_indices=set(interior_point_indices)-set(forbidden_interior_indices)
    print("Population size:", len(set_allowed_indices))
    list_allowed_indices=list(set_allowed_indices)
    
    
    
    # Median edges length for a aspecific number of points #
    target_edge_length=get_median_edge_length_population(nb_of_edges,nb_of_points)
    list_of_pairs=[]
    
    forbidden_neighbor_zone=target_edge_length*0.2
    
    
    # The list of allowed indices is the same for the one point sampling
    # Maybe reduce the number of grid points for faster sampling
    nb_of_samples=150
    list_of_pairs=[]
    count=0
    
    random.shuffle(list_allowed_indices)
    list_of_pairs=[]
    while count!=nb_of_samples:
        if nb_of_points !=1:
            random.shuffle(list_allowed_indices)
            for p in itertools.combinations(list_allowed_indices,nb_of_points):
    
                distance_preserved=[]
                for t in itertools.combinations(p,2):
                    distance=np.linalg.norm(grid_points[t[0]]-grid_points[t[1]])
                    if distance>forbidden_neighbor_zone:
                        distance_preserved.append(True)
                    else:
                        distance_preserved.append(False)
                if all(distance_preserved)==True:
    
                    list_of_pairs.append(p)
                    count+=1
                    break
        else:
            list_of_pairs.append(random.choice(list_allowed_indices))
            count+=1
    
    polygon_with_grid_points=[]
    for i in list_of_pairs:
        random_points=np.array(grid_points[np.asarray(i)])
        polygon_with_grid_points.append(np.vstack([polygon,random_points]))
        
    for polygon_with_grid_point in polygon_with_grid_points:
        polygon=polygon_with_grid_point[:nb_of_edges]
        inner_points=polygon_with_grid_point[nb_of_edges:]
        minimum_quality,mean_quality=quality_matrices(polygon,inner_points)
        with open(homedirectory+'/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/qualities_with_sampling/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons_with_grid_sampling_part2','ab') as f:
            pickle.dump(polygon_with_grid_point,f)
        with open(homedirectory+'/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/qualities_with_sampling/min_qualities_with_grid_sampling_part2','ab') as f:
            pickle.dump(minimum_quality,f)
        
    
                    
        
       
    
        
        

