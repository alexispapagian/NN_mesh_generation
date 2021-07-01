# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:50:21 2019

@author: papagian
"""


import sys
sys.path.insert(0, '../Triangulation/')
sys.path.insert(0, '../network_datasets')
sys.path.insert(0,'../point_coordinates_regression')




from Triangulation import *
import Triangulation_with_points

import torch
import torch.optim as optim
import struct



import torch.nn as nn

from torch.autograd import Variable
from math import atan2,pow,acos
from  Neural_network import *

from torch.autograd.function import Function
from point_coordinates_regression import *
from bilinear_interpolation_grid import select_points


import matplotlib.cm as cm
import matplotlib.colors as colors
from functools import reduce

import itertools
from torch.utils.data import Dataset,DataLoader

from Triangulation import *
import Triangulation_with_points



import numpy as np
from math import atan2,pow,acos

from point_coordinates_regression import *
from bilinear_interpolation_grid import select_points



from functools import reduce

import itertools


def loadall(filename):
    lis=[]
    with open(filename, "rb") as f:
        while True:
            try:
                lis.append(pickle.load(f))
            except EOFError:
                break
    return lis


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
    return np.median(merged)


def get_qualities_by_sector_updated(grid_points,inner_points,contour,nb_sectors,nb_of_edges,nb_of_points,outing_zone):
      
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
                        if np.linalg.norm(grid_point-contour_point)<outing_zone:
                            quality=1.3

                        one_third=0.25*contour[index]+0.75*contour[(index+1)%len(contour)]
                        two_thirds=0.75*contour[index]+0.25* contour[(index+1)%len(contour)]


                        mid_point=0.5*(contour[index]+contour[(index+1)%len(contour)])

                        one_fith=0.2*contour[index]+0.8*contour[(index+1)%len(contour)]
                        two_fifth=(2/5)*contour[index]+(3/5)*contour[(index+1)%len(contour)]
                        third_fifth=(3/5)*contour[index]+(2/5)*contour[(index+1)%len(contour)]
                        fourth_fith=0.8*contour[index]+0.2*contour[(index+1)%len(contour)]

                        one_sixth=(1/6)*contour[index]+(5/6)*contour[(index+1)%len(contour)]
                        two_sixth=(2/6)*contour[index]+(6/6)*contour[(index+1)%len(contour)]
                        fourth_sixth=(4/6)*contour[index]+(2/6)*contour[(index+1)%len(contour)]
                        fifth_sixth=(5/6)*contour[index]+(1/6)*contour[(index+1)%len(contour)]
                        condition1=np.linalg.norm(grid_point-one_sixth)<outing_zone or  np.linalg.norm(grid_point-two_sixth)<outing_zone or  np.linalg.norm(grid_point-fourth_sixth)<outing_zone or  np.linalg.norm(grid_point-fifth_sixth)<outing_zone
                        condition2=np.linalg.norm(grid_point-one_fith)<outing_zone or  np.linalg.norm(grid_point-two_fifth)<outing_zone or np.linalg.norm(grid_point-third_fifth)<outing_zone or  np.linalg.norm(grid_point-fourth_fith)<outing_zone or  np.linalg.norm(grid_point-fifth_sixth)<outing_zone

                        if  condition2 or condition1 or np.linalg.norm(grid_point-mid_point)<outing_zone or np.linalg.norm(grid_point-one_third)<outing_zone or np.linalg.norm(grid_point-two_thirds)<outing_zone:
                            quality=1.3
                    quality_point.append(quality)
            quality_sectors.append(quality_point) 
    return np.array(quality_sectors)


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



def l2_relative_error(A,B):
    A=np.array(A)
    B=np.array(B)
    diff=abs(B-A).flatten()
    maximum_error_index=np.argmax(diff)
    return np.linalg.norm(B-A)/np.linalg.norm(A),maximum_error_index


nb_of_edges=5
nb_of_points=1


nb_of_grid_points=20
X=np.linspace(-1.3,1.3,nb_of_grid_points)
Y=np.linspace(-1.3,1.3,nb_of_grid_points)
XX,YY=np.meshgrid(X,Y)

grid_points=np.array([[x,y] for x in X for y in Y])

polygons_filepath=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons.pkl'


grid_qualities_path=str(nb_of_edges)+'_polygons/'+str(nb_of_edges)+'_'+str(nb_of_points)+'_polygons/patch_scores/'
polygons_with_points=loadall(polygons_filepath)

polygons_with_points=np.array(polygons_with_points)
polygons=polygons_with_points[:,:2*nb_of_edges]
points=polygons_with_points[:,2*nb_of_edges:]

target_edge_length=get_median_edge_length_population(nb_of_edges,nb_of_points)

nb_sectors=int(nb_of_grid_points/2)# Grid will be divided into nb_sectors**2 patches

outing_zone=0.2*target_edge_length
polygons_qualities_sector=[]
for index,point in enumerate(points):
    contour=polygons[index].reshape(nb_of_edges,2)
    point=point.reshape(nb_of_points,2)
    grid_qualities=get_qualities_by_sector_updated(grid_points,point,contour,nb_sectors,nb_of_edges,nb_of_points,outing_zone)
    polygons_qualities_sector.append(grid_qualities)
polygons_qualities_sector=np.array(polygons_qualities_sector)
             
if not os.path.exists(grid_qualities_path):
    os.makedirs(grid_qualities_path)
grid_patch_scores_filepath=os.path.join(grid_qualities_path,str(nb_of_edges)+'_'+str(nb_of_points)+'_'+str(nb_of_grid_points)+'_sector_qualities.pkl')
with open(grid_patch_scores_filepath,'wb') as f:
    pickle.dump(polygons_qualities_sector ,f)
