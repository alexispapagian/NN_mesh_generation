
import numpy as np

import os
import copy

from itertools import permutations
from matplotlib import pyplot as plt
from numpy import pi,sin,cos,sqrt

import triangle as tri
import triangle.plot as plot
import time

from scipy.spatial import ConvexHull
from matplotlib.path import Path
from sklearn  import manifold

import pdb
import subprocess

import pickle

'''
  ======================================
  Main idea of finding the connectivity:
  ======================================
  
1) Build set of elements and edges
2) Sort edges according to maximum quality
3) for each pair of the edge put into set of edges  of the maximum quality
4) Put also to the set of elements the element that is formed
5) Proceed to the next edge and check if the edge formed from each pair already exist
6) If a pair of edges already exists proceed to next edge
7) If yes proceed to the next element

(*) Add function computing the quality of the mesh given that every point of a contour is connected 
    to a point of the mesh -> normalize the qualities for each point -> Add as parameter to neural network                                             Issues:

(1) Avoiding the consideration of triangles that are invalid ( by setting quality to 0):
     
     1st Approach (Failed):
     See if formed triangle contains points of the polygon
     
     2nd Approach :
     Calculate the angles of the polygon. Once that is done, any edge departing from a point in the polygon
     can have a greater angle from the departed edge greater than the edge has with the pre-existing edges
     
     
(2) Avoiding interseactions when creating elements:

    Here the main idea is whenever a new element is created to check if it includes a forabideen vertex. If it does 
    the new element can't be formed and we porceed to check the connection with the second smallest quality.
    
   ==========================================================================
      How to check if the vertex is locked ( So no new connections with it.)
   ==========================================================================
    
    
    So the idea is that given  a vertex we check the edges and the elements that include those edges.
    If start from an edge of the contour and end up to an edge of the contour again then the vertex
    
    
    + For every vertex of the element that is to be created  and
        
        for vtx in element:
        
            if closed_ring(vtx,adj_vertices,elements):
            
                  don't created element
                  proceed to connection with second most great quality
                  if doesn't exist:
                  proceeed  to next edge
                  
    + for the vtx and the adjacent v1 look for the element 
        (vtx,v1,v2) ->  (vtx, v2) -> (vtx,v3,v2) -> (vtx,v3)-> ... -> (vtx,vn) 
        Check if end is edge of contour if yes then vtx is forbidden -> insert to forbidden vertices
                
      '''         
     
     
# In[3]:

def BCE_accuracy(model,variable,labels):
    net.eval()
    predictions=model(variable).data.numpy()
    predictions[np.where(predictions>0.5)]=1
    predictions[np.where(predictions<=0.5)]=0
    diff=labels-predictions
    correct_prediction=0
    for i in diff:
        if (not i.any()):
            correct_prediction+=1
    net.train()
    return  100*correct_prediction/variable.size()[0],diff




def connectivity_information(triangulation,print_info=False):
    
    segments= tuple(triangulation['segments'].tolist())
    triangles=tuple(triangulation['triangles'].tolist())
    vertices=triangulation['vertices']   
    
    connect_info={str(r):[0 for i in range(len(vertices))] for r in tuple(triangulation['segments'].tolist())}
    for segment in segments:
        for triangle in triangles:
            if set(segment).issubset(set(triangle)):
                connection=set(triangle)-set(segment)
                if print_info: print("segment:",segment,"is connected to:",connection,"to form triangle:",triangle)
                connect_info[str(segment)][tuple(connection)[0]]=1    
    return connect_info



def get_labels(triangulation,connect_info):
    indices=[]
    vertices=list(range(triangulation['vertices'].shape[0]))
    for i in triangulation['segments']:
           indices.append(set(vertices)-set(i)) 
    labels=[]
    list_values=list(connect_info.values())
    for i in range(len(list_values)):
        for j in indices[i]:
            labels.append(list_values[i][j])
    return  labels



def rot(theta):
    return np.array([[cos(theta),-sin(theta)],     
                     [sin(theta),cos(theta)]])




def get_reference_polygon(nb_of_points,plot=False):
    angles=np.empty(nb_of_points)
    points=np.empty([nb_of_points,2])
    plot_coords=np.empty([nb_of_points,2])
    indices=[]
    angle_division=2*pi/nb_of_points
    
   
    for i in range(nb_of_points):
        angle=i*angle_division
        angles[i]=angle
        point=np.array([1,0]) #pick edge length of 1
        points[i]=np.dot(rot(angle),point.T)  #rotate it according to the  chosen angle
        indices.append(i)
   
    if plot==True:
        plot_coords=np.vstack([points,points[0]])
        (s,t)=zip(*plot_coords)
        plt.plot(s,t)
        for index,i in enumerate(indices):
            plt.annotate(str(i),(s[index],t[index]))
    
    return points



def generate_contour(nb_of_points,plot=False):
    
    
    angles=np.empty(nb_of_points)
    points=np.empty([nb_of_points,2])
    plot_coords=np.empty([nb_of_points,2])
    indices=[]
    angle_division=2*pi/nb_of_points
   
    for i in range(nb_of_points):
        angle=((i+1)*angle_division-i*angle_division)*np.random.random_sample()+i*angle_division
        angles[i]=angle
        point=np.array([np.random.uniform(0.3,1),0]) #pick random point at (1,0)
       #point=np.array([1,0]) #pick edge length of 1

        points[i]=np.dot(rot(angle),point.T)  #rotate it according to the  chosen angle
        indices.append(i)
   
    if plot==True:
        plot_coords=np.vstack([points,points[0]])
        (s,t)=zip(*plot_coords)
        plt.plot(s,t)
        for index,i in enumerate(indices):
            plt.annotate(str(i),(s[index],t[index]))
    
    return points


def plot_contour(contour):    
    plot_coords=np.vstack([contour,contour[0]])
    (s,t)=zip(*plot_coords)
    plt.plot(s,t)
    indices=[i for i in range(contour.shape[0])]
    for index,i in enumerate(indices):
        plt.annotate(str(i),(s[index],t[index]))
    
    
def convert2geo(contour,target_edge_length):
    filepath="..\\gmsh\\"
    with open(os.path.join(filepath,'contour.geo'),'w') as file_input:
        
        file_input.write("SetFactory(\" OpenCASCADE \" );\n")
        file_input.write("// Coordinate points\n")
        for i in range(contour.shape[0]):
            file_input.write('Point({}) = {} {},{},{},{} {} ;\n'.format(i+1,'{',contour[i][0],contour[i][1],0,1,'}'))
            
        file_input.write("// Segments\n")
        for i in range(contour.shape[0]):
            if i+2==contour.shape[0]+1:
                file_input.write('Line({})={} {},{} {} ;\n'.format(i+1,'{',i+1,1,'}'))
                break
            file_input.write('Line({})={} {},{} {} ;\n'.format(i+1,'{',i+1,i+2,'}'))
            
            
        file_input.write("// LineLoop\n")
        file_input.write("Line Loop({})={}".format(1,'{'))
        for i in range(contour.shape[0]):
            if i ==contour.shape[0]-1:
                file_input.write('{}'.format(i+1))
                break
            file_input.write('{},'.format(i+1))
        file_input.write('};\n')
            
            
        file_input.write("// Surface\n")
        file_input.write("Plane Surface({})={} {} {} ;".format(1,'{',1,'}'))
        file_input.write("// Add transfinite lines\n")
        for i in range(contour.shape[0]):
            file_input.write("Transfinite Line {{{}}}=1 Using Bump 1;\n".format(i+1))
        file_input.write("// Target edge length\n")
        file_input.write('Mesh.CharacteristicLengthFactor={};'.format(target_edge_length))
        
        
        
    
def convert2geo_additional(contour,target_edge_length):
    filepath="..\\gmsh_additional\\"
    with open(os.path.join(filepath,'contour.geo'),'w') as file_input:
        
        file_input.write("SetFactory(\" OpenCASCADE \" );\n")
        file_input.write("// Coordinate points\n")
        for i in range(contour.shape[0]):
            file_input.write('Point({}) = {} {},{},{},{} {} ;\n'.format(i+1,'{',contour[i][0],contour[i][1],0,1,'}'))
            
        file_input.write("// Segments\n")
        for i in range(contour.shape[0]):
            if i+2==contour.shape[0]+1:
                file_input.write('Line({})={} {},{} {} ;\n'.format(i+1,'{',i+1,1,'}'))
                break
            file_input.write('Line({})={} {},{} {} ;\n'.format(i+1,'{',i+1,i+2,'}'))
            
            
        file_input.write("// LineLoop\n")
        file_input.write("Line Loop({})={}".format(1,'{'))
        for i in range(contour.shape[0]):
            if i ==contour.shape[0]-1:
                file_input.write('{}'.format(i+1))
                break
            file_input.write('{},'.format(i+1))
        file_input.write('};\n')
            
            
        file_input.write("// Surface\n")
        file_input.write("Plane Surface({})={} {} {} ;".format(1,'{',1,'}'))
        file_input.write("// Add transfinite lines\n")
        for i in range(contour.shape[0]):
            file_input.write("Transfinite Line {{{}}}=1 Using Bump 1;\n".format(i+1))
        file_input.write("// Target edge length\n")
        file_input.write('Mesh.CharacteristicLengthFactor={};'.format(target_edge_length))

def convert2geo_jupyter(contour,target_edge_length):
    filepath="..\\gmsh_jupyter\\"
    with open(os.path.join(filepath,'contour.geo'),'w') as file_input:
        
        file_input.write("SetFactory(\" OpenCASCADE \" );\n")
        file_input.write("// Coordinate points\n")
        for i in range(contour.shape[0]):
            file_input.write('Point({}) = {} {},{},{},{} {} ;\n'.format(i+1,'{',contour[i][0],contour[i][1],0,1,'}'))
            
        file_input.write("// Segments\n")
        for i in range(contour.shape[0]):
            if i+2==contour.shape[0]+1:
                file_input.write('Line({})={} {},{} {} ;\n'.format(i+1,'{',i+1,1,'}'))
                break
            file_input.write('Line({})={} {},{} {} ;\n'.format(i+1,'{',i+1,i+2,'}'))
            
            
        file_input.write("// LineLoop\n")
        file_input.write("Line Loop({})={}".format(1,'{'))
        for i in range(contour.shape[0]):
            if i ==contour.shape[0]-1:
                file_input.write('{}'.format(i+1))
                break
            file_input.write('{},'.format(i+1))
        file_input.write('};\n')
            
            
        file_input.write("// Surface\n")
        file_input.write("Plane Surface({})={} {} {} ;".format(1,'{',1,'}'))
        file_input.write("// Add transfinite lines\n")
        for i in range(contour.shape[0]):
            file_input.write("Transfinite Line {{{}}}=1 Using Bump 1;\n".format(i+1))
        file_input.write("// Target edge length\n")
        file_input.write('Mesh.CharacteristicLengthFactor={};'.format(target_edge_length))
            
def call_gmsh(algorithm='meshadapt'):
    filepath="..\\gmsh\\"
    try:
        #subprocess.Popen(['-2',os.path.join(filepath,"contour.geo"),' -algo','meshadapt','-o','contour_mesh.msh'],executable=os.path.join(filepath,"gmsh.exe"),shell=False)
        subprocess.call([os.path.join(filepath,'gmsh.exe'),os.path.join(filepath,'contour.geo'),'-2','-algo',algorithm,'-o',os.path.join(filepath,'output.msh')],shell=False)
        # os.system('D:\\Users\\papagian\\python\\Dimensionality_reduction\\gmsh\\gmsh -2 contour.geo')
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        

def call_gmsh_additional(algorithm='meshadapt'):
    filepath="..\\gmsh_additional\\"
    try:
        #subprocess.Popen(['-2',os.path.join(filepath,"contour.geo"),' -algo','meshadapt','-o','contour_mesh.msh'],executable=os.path.join(filepath,"gmsh.exe"),shell=False)
        subprocess.call([os.path.join(filepath,'gmsh.exe'),os.path.join(filepath,'contour.geo'),'-2','-algo',algorithm,'-o',os.path.join(filepath,'output.msh')],shell=False)
        # os.system('D:\\Users\\papagian\\python\\Dimensionality_reduction\\gmsh\\gmsh -2 contour.geo')
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))


def call_gmsh_jupyter(algorithm='meshadapt'):
    filepath="..\\gmsh_jupyter\\"
    try:
        #subprocess.Popen(['-2',os.path.join(filepath,"contour.geo"),' -algo','meshadapt','-o','contour_mesh.msh'],executable=os.path.join(filepath,"gmsh.exe"),shell=False)
        subprocess.call([os.path.join(filepath,'gmsh.exe'),os.path.join(filepath,'contour.geo'),'-2','-algo',algorithm,'-o',os.path.join(filepath,'output.msh')],shell=False)
        # os.system('D:\\Users\\papagian\\python\\Dimensionality_reduction\\gmsh\\gmsh -2 contour.geo')
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        
def read_mesh(contour):
        filepath="..\\gmsh\\"
        with open(os.path.join(filepath,'output.msh'),'r') as mesh_output:
            found_points=False
            found_nb_points=False
            extra_points=[]
            lines = filter(None, (line.rstrip() for line in mesh_output)) # not taking into account empty spaces
            for line in lines:
                if '$Nodes'  in line:
                    found_points=True
                    found_nb_points=True
                    line=mesh_output.readline()
                    #print(line)
                    
                if found_points:

                    if'$EndNodes' in line:
                        found_points=False
                        mesh_output.close()
                        break
                    if found_nb_points:
                        for i in line.split():
                            nb_extra_points=int(i)-contour.shape[0]
                          #  print("found {} extra points".format(nb_extra_points))
                            found_nb_points=False
                            line=mesh_output.readline()
                    vertex_index=line.split()[0]
                    x_coord=line.split()[1]
                    y_coord=line.split()[2]
#                    z_coord=line.split()[3]
                    if int(vertex_index) > contour.shape[0]:         
                       print(line)
                       point=np.array([float(x_coord),float(y_coord)])
                       extra_points.append(point)
        return nb_extra_points,extra_points


def read_mesh_additional(contour):
        filepath="..\\gmsh_additional\\"
        with open(os.path.join(filepath,'output.msh'),'r') as mesh_output:
            found_points=False
            found_nb_points=False
            extra_points=[]
            lines = filter(None, (line.rstrip() for line in mesh_output)) # not taking into account empty spaces
            for line in lines:
                if '$Nodes'  in line:
                    found_points=True
                    found_nb_points=True
                    line=mesh_output.readline()
                    #print(line)
                    
                if found_points:

                    if'$EndNodes' in line:
                        found_points=False
                        mesh_output.close()
                        break
                    if found_nb_points:
                        for i in line.split():
                            nb_extra_points=int(i)-contour.shape[0]
                          #  print("found {} extra points".format(nb_extra_points))
                            found_nb_points=False
                            line=mesh_output.readline()
                    vertex_index=line.split()[0]
                    x_coord=line.split()[1]
                    y_coord=line.split()[2]
#                    z_coord=line.split()[3]
                    if int(vertex_index) > contour.shape[0]:         
                       print(line)
                       point=np.array([float(x_coord),float(y_coord)])
                       extra_points.append(point)
        return nb_extra_points,extra_points
    

def read_mesh_jupyter(contour):
        filepath="..\\gmsh_jupyter\\"
        with open(os.path.join(filepath,'output.msh'),'r') as mesh_output:
            found_points=False
            found_nb_points=False
            extra_points=[]
            lines = filter(None, (line.rstrip() for line in mesh_output)) # not taking into account empty spaces
            for line in lines:
                if '$Nodes'  in line:
                    found_points=True
                    found_nb_points=True
                    line=mesh_output.readline()
                    #print(line)
                    
                if found_points:

                    if'$EndNodes' in line:
                        found_points=False
                        mesh_output.close()
                        break
                    if found_nb_points:
                        for i in line.split():
                            nb_extra_points=int(i)-contour.shape[0]
                          #  print("found {} extra points".format(nb_extra_points))
                            found_nb_points=False
                            line=mesh_output.readline()
                    vertex_index=line.split()[0]
                    x_coord=line.split()[1]
                    y_coord=line.split()[2]
#                    z_coord=line.split()[3]
                    if int(vertex_index) > contour.shape[0]:         
                       print(line)
                       point=np.array([float(x_coord),float(y_coord)])
                       extra_points.append(point)
        return nb_extra_points,extra_points
    
    
    

def get_triangle_indices(contour):
        filepath="..\\gmsh\\"
        with open(os.path.join(filepath,'output.msh'),'r') as mesh_output:
            
            lines = filter(None, (line.rstrip() for line in mesh_output)) # not taking into account empty spaces
            found_elements=False
            indices=[]
            triangle_indices=[]
            real_triangles=[]
            for line in lines:
                if '$Elements'  in line and not found_elements:
                    found_elements=True
                    line=mesh_output.readline().rstrip() 
                    line=mesh_output.readline().rstrip()    

                    

                   # print(line)
                   # line=mesh_output.readline()    
                    
                if found_elements:
                     if '$EndElements'in line:
                         found_elements=False
                         break
                    # print(line)
                     indices.append(line)
                 
            for numbers in indices:
                for number in numbers:
                    numbers=[index.strip() for  index in number.split(',')]
            
            for i in indices:
                triangle_indices.append(i.split(' ')[::-1][0:3])
            
            for i,_ in enumerate(triangle_indices):
                 for j,o in enumerate(_):
                      triangle_indices[i][j]=int(triangle_indices[i][j])-1
                      
            for i in triangle_indices:
                 if(len(set(i)))==3:
                     real_triangles.append(i)
                
        return real_triangles
    
    
    
    

        return real_triangles
    
    
    
def get_triangles(contour,indices,points):
    points=np.array(points)
    if len(points)>0:
        contour_with_points=np.vstack([contour,points])
    else:
        contour_with_points=contour
        
    triangles=[]
    for i in indices:
        triangle=contour_with_points[i]
        triangles.append( triangle)
    return triangles
 
def get_extrapoints_target_length(contour,target_edge_length,algorithm='meshadapt'):
    convert2geo(contour,target_edge_length)
    call_gmsh(algorithm)
    nb_of_points,point_coords=read_mesh(contour)
    return nb_of_points,point_coords
                    
def get_reference_time(contour,target_edge_length,algorithm='meshadapt'):
    convert2geo(contour,target_edge_length)
    time_start=time.clock()
    call_gmsh(algorithm)
    return  time.clock()-time_start

 
def get_extrapoints_target_length_additional(contour,target_edge_length,algorithm='meshadapt'):
    convert2geo_additional(contour,target_edge_length)
    call_gmsh_additional(algorithm)
    nb_of_points,point_coords=read_mesh_additional(contour)
    return nb_of_points,point_coords

def get_extrapoints_target_length_jupyter(contour,target_edge_length,algorithm='meshadapt'):
    convert2geo_jupyter(contour,target_edge_length)
    call_gmsh_jupyter(algorithm)
    nb_of_points,point_coords=read_mesh_jupyter(contour)
    return nb_of_points,point_coords
 
def get_extrapoints_target_length2(contour,target_edge_length,algorithm='meshadapt'):
    convert2geo2(contour,target_edge_length)
    call_gmsh2(algorithm)
    nb_of_points,point_coords=read_mesh2(contour)
    return nb_of_points,point_coords
                    

def save_dataset(filename,dataset):
    path=os.path.join('../polygon_datasets/',filename)
    
    with open(path,'wb') as output:
        pickle.dump(dataset,output)
        
def load_dataset(filename):

    path=os.path.join('../polygon_datasets/',filename)
    with open(path,'rb') as input:
        dataset=pickle.load(input)
    return dataset                    
                    
                    
    

def apply_procrustes(polygon_points,ref_polygon=None,plot=False):  
    
    # Get reference polygona and adjust any random polygon to that
    if ref_polygon is None:    
        ref_polygon=get_reference_polygon(polygon_points.shape[0])
    
    
    #Mean of each coordinate
    mu_polygon = polygon_points.mean(0)
    mu_ref_polygon = ref_polygon.mean(0)
    
    #Centralize data to the mean 
    centralised_ref_polygon_points = ref_polygon-mu_ref_polygon
    centralised_polygon_points = polygon_points-mu_polygon
    
    #Squared sum of X-mean(X)
    ss_ref_polygon_points = (centralised_ref_polygon_points**2.).sum()
    ss_polygon_points = (centralised_polygon_points**2.).sum()

       
    #Frobenius norm of X
    norm_ss_ref_polygon_points = np.sqrt(ss_ref_polygon_points)
    norm_ss_polygon_points = np.sqrt(ss_polygon_points)

    
    # scale to equal (unit) norm
    centralised_ref_polygon_points /=norm_ss_ref_polygon_points     
    centralised_polygon_points /=norm_ss_polygon_points
        
    
    #Finding best rotation to superimpose on regular triangle
    #Applying SVD to the  matrix 
    A = np.dot(centralised_ref_polygon_points.T, centralised_polygon_points)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V=Vt.T
    R = np.dot(V,U.T)
    
  
    traceTA = s.sum()
    indices=[i for i in range(polygon_points.shape[0])]
    
   

    polygon_transformed =norm_ss_ref_polygon_points*traceTA*np.dot(centralised_polygon_points,R)+mu_ref_polygon

    if plot==True:
        plot_coords=np.vstack([polygon_transformed,polygon_transformed[0]])
        (s,t)=zip(*plot_coords)
        plt.plot(s,t)
        for index,i in enumerate(indices):
            plt.annotate(str(i),(s[index],t[index]))
    
    return polygon_transformed
        
    

def rotation_projection(nb_of_points):
        
    contours=[]
    ref_polygon=get_reference_polygon(nb_of_points,True)
    random_contour=generate_contour(nb_of_points,True)
    contours.append(ref_polygon)
    
    contours.append(random_contour)
    
    for i in range(1,nb_of_points):    
        rotated_contour=np.dot(rot(pi/i),random_contour.T).T
        plot_contour(rotated_contour)
        contours.append(rotated_contour)
    
    contours=np.array(contours)
    
    # Projecting via Isomap 
    
    contours=contours.reshape(contours.shape[0],2*nb_of_points)
    isomap=manifold.Isomap(n_neighbors=2,n_components=2)
    isomap.fit(contours)
    Polygons_manifold_2D=isomap.transform(contours)
    Polygons_manifold_2D=isomap.transform(contours)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(Polygons_manifold_2D[:,0],Polygons_manifold_2D[:,1],color=['red' if i is 0 else 'blue'
                                                                      for i,_ in enumerate(Polygons_manifold_2D)])
    ax.set_title('Isomap projection without procrustes')
    ax.set_xlabel('1st Component')
    ax.set_ylabel('2nd Component')
    
    # Applying procrustes
    procrustes_points=np.empty([contours.shape[0],nb_of_points,2])
    for i,contour in enumerate(contours):
        procrustes_points[i]=apply_procrustes(contour,True)
    ref_polygon=get_reference_polygon(nb_of_points,True)
    
def scale_projection(polygon):
    pass
    




        
  


# In[25]:


def contains_points(triangle,polygon):
    try:
        hull=ConvexHull(triangle)
    except:
        print("Invalid Convex hull")
        return True
    hull_path=Path(triangle[hull.vertices])
    set_polygon=set(tuple(i) for i in polygon)
    set_triangle=set(tuple(i) for i in triangle)
    #print(set_polygon,set_triangle)
    difference=set_polygon-set_triangle
    
    if len(difference)==0:
        return False

    for i in difference:
        if hull_path.contains_point(i):
            return True
            break
    return False


def is_counterclockwise(polygon):  
    area = 0
    counterclokwise=False
    for index,_ in enumerate(polygon):
        second_index=(index+1)%len(polygon)
        area+=polygon[index][0]*polygon[second_index][1]
        area-=polygon[second_index][0]*polygon[index][1]
    if area/2<0:
        counterclokwise=False
    else:
        counterclokwise=True
    return counterclokwise



def compute_edge_lengths(pt1,pt2):
    return np.linalg.norm(pt1-pt2)

def compute_edge_lengths2(triangle):
    edgelengths2=np.empty([2,3])
    for i in range(2):
        for j in range(i+1,3):
            eij=triangle[j]-triangle[i]
            edgelengths2[i][j]=np.dot(eij,eij)
    return edgelengths2
    

def compute_triangle_edge_lengths(triangle):
    edge_lengths=[length for i in range(3) for length in compute_edge_lengths(triangle[int(i)],triangle[int((i+1)%3)]) ]
    return edge_lengths

def compute_triangle_normals(triangle):
   
    e01=triangle[1]-triangle[0]
    e02=triangle[2]-triangle[0]
    
    e01_cross_e02=np.cross(e01,e02)
    
    return e01_cross_e02


def compute_triangle_barycenter(triangle):
    barycenter=np.array([triangle[:,0].sum()/3,triangle[:,1].sum()/3])
    return barycenter
    


def compute_triangle_area(triangle):
   
    e01=triangle[1]-triangle[0]
    e02=triangle[2]-triangle[0]
    
    e01_cross_e02=np.cross(e01,e02)
    
    # Omit triangles that are inverted (out of the domain)
    if e01_cross_e02<0:
        return 0
        
    
    e01_cross_e02_norm=np.linalg.norm(e01_cross_e02)

        
    return e01_cross_e02_norm/2


def compute_minimum_quality_triangulated_contour(polygon,element_list):
    triangle_qualities=[]
    for element in element_list:
        indices=np.asarray(element)
        triangle=polygon[indices]
        
        sum_edge_lengths=0
        edge_length2=compute_edge_lengths2(triangle)
        for i in range(2):
            for j in range(i+1,3):
                sum_edge_lengths+=edge_length2[i][j]
    
    
        factor=4/sqrt(3)
        e01=triangle[1]-triangle[0]
        e02=triangle[2]-triangle[0]
    
        e01_cross_e02=np.cross(e01,e02)
        
        area=np.linalg.norm(e01_cross_e02)/2
    
        lrms=sqrt(sum_edge_lengths/3)
        lrms2=lrms**2
        quality=(area/lrms2)*factor
        
        triangle_qualities.append(quality)
        
    
    triangle_qualities=np.array(triangle_qualities)
    return triangle_qualities.min()

def compute_mean_quality_triangulated_contour(polygon,element_list):
    triangle_qualities=[]
    for element in element_list:
        indices=np.asarray(element)
        triangle=polygon[indices]
        
        sum_edge_lengths=0
        edge_length2=compute_edge_lengths2(triangle)
        for i in range(2):
            for j in range(i+1,3):
                sum_edge_lengths+=edge_length2[i][j]
    
    
        factor=4/sqrt(3)
        e01=triangle[1]-triangle[0]
        e02=triangle[2]-triangle[0]
    
        e01_cross_e02=np.cross(e01,e02)
        
        area=np.linalg.norm(e01_cross_e02)/2
    
        lrms=sqrt(sum_edge_lengths/3)
        lrms2=lrms**2
        quality=(area/lrms2)*factor
        
        triangle_qualities.append(quality)
        
    
    triangle_qualities=np.array(triangle_qualities)
    return triangle_qualities.mean()

def compute_triangle_quality(triangle,polygon=None):
    
    if polygon is None:
        polygon=triangle
    
    # The incoming triangle has edged [p0,p1] which is an edge and p2 is the connection
    polygon_angles=get_polygon_angles(polygon)
    
    indices=[]


    for point in triangle: 
        for index,point_in_polygon in enumerate(polygon):
              if np.allclose(point,point_in_polygon):
                    indices.append(index)
                
    p0,p1,p2=indices[0],indices[1],indices[2]
    
    neighbor_points=connection_indices(p2,get_contour_edges(polygon))
    
    # Checking if edges of connected poiints form an angle bigger than the polygon angles
    if (polygon_angles[p0]<calculate_angle(polygon[p0],polygon[p1],polygon[p2]) 
        or polygon_angles[p1]<calculate_angle(polygon[p1],polygon[p0],polygon[p2])):
        #print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        return 0
    
    
    if( polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[0]],polygon[p0])
        or polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[0]],polygon[p1])
    ):
        #print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        return 0
    
    if( polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[1]],polygon[p0])
        or polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[1]],polygon[p1])):
        #print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        return 0
    
    factor=4/sqrt(3)
    area=compute_triangle_area(triangle)
   
    if area==0:
        return 0
    
    if contains_points(triangle,polygon):
        return 0
    
    
    
    #edgelengths=np.empty([triangle.shape[0],1])
    
    #for i,_ in enumerate(triangle):
     #   edgelengths[i]=compute_edge_length(triangle[i],triangle[(i+1)%3])
    
    #sum_edge_lengths=edgelengths.sum() 
    #factor=4/sqrt(3)*3/sum_edge_lengths
    
    sum_edge_lengths=0
    edge_length2=compute_edge_lengths2(triangle)
    for i in range(2):
        for j in range(i+1,3):
            sum_edge_lengths+=edge_length2[i][j]
    
    
    
    lrms=sqrt(sum_edge_lengths/3)
    lrms2=lrms**2
    quality=area/lrms2
    
    return quality*factor


def compute_center_of_mass(triangles):

    areas=[]
    barycenters=[]
    
    for triangle in triangles:
        barycenters.append(compute_triangle_barycenter(triangle))
        areas.append(compute_triangle_area(triangle))
    
    barycenters=np.array(barycenters)
    areas=np.array(areas)
    
    areas_times_barycenters=np.multiply(areas.reshape(len(areas),1),barycenters)
    
    return np.array( areas_times_barycenters.sum(0)/areas.sum())


def compute_delaunay_minimum_quality(polygon):
        triangles_in_mesh=[]
        contour_connectivity=get_contour_edges(polygon)
        shape=dict(vertices=polygon,segments=contour_connectivity)
        
        t = tri.triangulate(shape, 'pq0')
        
        for triangle_index in t['triangles']:
            triangles_in_mesh.append(polygon[np.asarray([triangle_index])])
        
        triangle_qualities=[]
        for triangle in triangles_in_mesh:
            triangle.resize(3,2)
            triangle_quality=compute_triangle_quality(triangle)
            triangle_qualities.append(triangle_quality)
        
        triangle_qualities=np.array(triangle_qualities)

        minimum_quality=triangle_qualities.min()
        
        
        return minimum_quality







def compute_minimum_quality_triangle(triangle,polygon=None):
    if polygon is None:
        polygon=triangle
    
    # The incoming triangle has edged [p0,p1] which is an edge and p2 is the connection
    polygon_angles=get_polygon_angles(polygon)
    
    indices=[]


    for point in triangle: 
        for index,point_in_polygon in enumerate(polygon):
              if np.allclose(point,point_in_polygon):
                    indices.append(index)
                
    p0,p1,p2=indices[0],indices[1],indices[2]
    
    neighbor_points=connection_indices(p2,get_contour_edges(polygon))
    
    # Checking if edges of connected poiints form an angle bigger than the polygon angles
    if (polygon_angles[p0]<calculate_angle(polygon[p0],polygon[p1],polygon[p2]) 
        or polygon_angles[p1]<calculate_angle(polygon[p1],polygon[p0],polygon[p2])):
        #print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        return 0
    
    
    if( polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[0]],polygon[p0])
        or polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[0]],polygon[p1])
    ):
        #print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        return 0
    
    if( polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[1]],polygon[p0])
        or polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[1]],polygon[p1])):
        #print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        return 0
    
    area=compute_triangle_area(triangle)
   
    if area==0:
        return 0
    
    ContainsPoints=False
	
    try:
        ContainsPoints=contains_points(triangle,polygon)
    except QhullError:
        ContainsPoints=True
    if ContainsPoints:
        return 0
    
    if contains_points(triangle,polygon):
        return 0
    
    triangles_in_mesh=[]
    triangles_in_mesh.append(triangle)
    contour_connectivity=get_contour_edges(polygon)
    contour_connectivity=np.vstack([contour_connectivity,[p0,p2],[p1,p2]])
    hole=np.array([(triangle.sum(0))/3])
    shape=dict(holes=hole,vertices=polygon,segments=contour_connectivity)
    t = tri.triangulate(shape, 'pq0')
    
    Invalid_triangulation=False

    try:   
        for triangle_index in t['triangles']:
            triangles_in_mesh.append(polygon[np.asarray([triangle_index])])
    except :
        print("Invalid triangulation",p0,p1,p2)
        Invalid_triangulation=True
        
    triangle_qualities=[]
    for triangle in triangles_in_mesh:
        triangle.resize(3,2)
        triangle_quality=compute_triangle_quality(triangle)
        triangle_qualities.append(triangle_quality)
    
    if Invalid_triangulation:
        mean_quality,minimum_quality=0,0
    else:
        triangle_qualities=np.array(triangle_qualities)
        mean_quality=triangle_qualities.mean()
        minimum_quality=triangle_qualities.min()

    return minimum_quality



def get_indices(triangle,polygon):
    
    indices=[]

    for point in triangle: 
        for index,point_in_polygon in enumerate(polygon):
              if np.allclose(point,point_in_polygon):
                    indices.append(index)
                    
    return indices
                    
                    

def compute_mean_quality_triangle(triangle,polygon=None):
    if polygon is None:
        polygon=triangle
    
    # The incoming triangle has edged [p0,p1] which is an edge and p2 is the connection
    polygon_angles=get_polygon_angles(polygon)
    
    indices=[]


    for point in triangle: 
        for index,point_in_polygon in enumerate(polygon):
              if np.allclose(point,point_in_polygon):
                    indices.append(index)
                
    p0,p1,p2=indices[0],indices[1],indices[2]
    
    neighbor_points=connection_indices(p2,get_contour_edges(polygon))
    
    # Checking if edges of connected poiints form an angle bigger than the polygon angles
    if (polygon_angles[p0]<calculate_angle(polygon[p0],polygon[p1],polygon[p2]) 
        or polygon_angles[p1]<calculate_angle(polygon[p1],polygon[p0],polygon[p2])):
        #print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        return 0
    
    
    if( polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[0]],polygon[p0])
        or polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[0]],polygon[p1])
    ):
        #print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        return 0
    
    if( polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[1]],polygon[p0])
        or polygon_angles[p2]<calculate_angle(polygon[p2],polygon[neighbor_points[1]],polygon[p1])):
        #print("Spotted inverted triangle: {}".format([p0,p1,p2]))
        return 0
    
    area=compute_triangle_area(triangle)
   
    if area==0:
        return 0
    
    if contains_points(triangle,polygon):
        return 0
    
    triangles_in_mesh=[]
    triangles_in_mesh.append(triangle)
    contour_connectivity=get_contour_edges(polygon)
    contour_connectivity=np.vstack([contour_connectivity,[p0,p2],[p1,p2]])
    hole=np.array([(triangle.sum(0))/3])
    shape=dict(holes=hole,vertices=polygon,segments=contour_connectivity)
    t = tri.triangulate(shape, 'pq0')
    
    Invalid_triangulation=False

    try:   
        for triangle_index in t['triangles']:
            triangles_in_mesh.append(polygon[np.asarray([triangle_index])])
    except :
        print("Invalid triangulation",p0,p1,p2)
        Invalid_triangulation=True
        
    triangle_qualities=[]
    for triangle in triangles_in_mesh:
        triangle.resize(3,2)
        triangle_quality=compute_triangle_quality(triangle)
        triangle_qualities.append(triangle_quality)
    
    if Invalid_triangulation:
        mean_quality=0,0
    else:
        triangle_qualities=np.array(triangle_qualities)
        mean_quality=triangle_qualities.mean()

    return mean_quality


# Quality of elements formed by connecting each edge with one of the other points of the contour

def quality_matrix(polygon,compute_minimum=True ,normalize=False):
    #polygon=apply_procrustes(polygon,False)

    contour_connectivity=np.array(list(tuple(i) for i in get_contour_edges(polygon)))
    
    
    
    quality_matrix=np.zeros([contour_connectivity.shape[0],polygon.shape[0]])
    #area_matrix=np.zeros([contour_connectivity.shape[0],polygon.shape[0]])
    normals_matrix=np.zeros([contour_connectivity.shape[0],polygon.shape[0]])

    list_of_triangles=[]
    
    for index,edge in enumerate(contour_connectivity):
        # Not omitting non triangles because either way their quality is zero
        triangles_to_edge_indices=[[*edge,i] for i in range(polygon.shape[0]) ]
        
        

        #print(triangles_to_edge_indices)
        triangles_to_edge_indices=np.asarray(triangles_to_edge_indices)
        triangles=polygon[triangles_to_edge_indices]
        list_of_triangles.append(triangles)
        
    
    list_of_triangles=np.array(list_of_triangles)
    
    if compute_minimum:
        for i,triangles in enumerate(list_of_triangles):
            for j,triangle in enumerate(triangles):
                quality_matrix[i,j]=compute_minimum_quality_triangle(triangle,polygon)
    else:
         for i,triangles in enumerate(list_of_triangles):
            for j,triangle in enumerate(triangles):
                quality_matrix[i,j]=compute_mean_quality_triangle(triangle,polygon)
    
            #area_matrix[i,j]=compute_triangle_area(triangle)
            #normals_matrix[i,j]=compute_triangle_normals(triangle)

            
    
    sum_of_qualities=quality_matrix.sum(1)

    if normalize is True:
        for i,_ in enumerate(quality_matrix):
            quality_matrix[i]/=sum_of_qualities[i]
    
    return quality_matrix,normals_matrix





def check_edge_validity(edge,polygon,set_edges,interior_edges):        
    # Check if new edges are already in the set
    found_in_set=False
    found_in_interior_set=False
    for index in range(len(polygon)):
        occuring_index=index

        edge1,edge2=tuple(permutations((edge[0],index))),tuple(permutations((edge[1],index)))
        condition1= edge1[0] in set_edges or edge1[1] in set_edges
        condition2= edge2[0] in set_edges or edge2[1] in set_edges
        condition3= edge1[0] in interior_edges or edge1[1] in interior_edges
        condition4= edge2[0] in interior_edges or edge2[1] in interior_edges
    
        
            # both edges are found in the list of set of edges (Invalid)
        if (condition1 and condition2): 
            found_in_set=True
            occuring_index=index
            
        
        # both edges are found in the list of interior edges created
        if (condition3 and condition4):
            found_in_interior_set=True
            occuring_index=index
            
        if found_in_interior_set or found_in_set:
            break
    return found_in_interior_set,found_in_set,occuring_index

def triangulate(polygon,ordered_quality_matrix,recursive=True):
    set_edges=set(tuple(i) for i in get_contour_edges(polygon))
    interior_edges=set()
    set_elements=set()
    set_locked_vertices=set()
    set_forbidden_intersections=set()
    
    print("initial set edges:", set_edges)
    


    for edge in ordered_quality_matrix.keys():
        
        found_in_interior_set,found_in_set,index=check_edge_validity(edge,polygon,set_edges,interior_edges)
        
        for qualities_with_edges in ordered_quality_matrix[edge][0]:
            
            element_created=False
           
            target_vtx=qualities_with_edges[1]
            
            if target_vtx==edge[0] or target_vtx==edge[1]:
                continue
           
            print("Edge:",edge,"targeting:",target_vtx)
        
            if found_in_interior_set:
                element=(edge[0],edge[1],index)  
                set_elements.add(element)
                print("Element inserted:",element)
                continue
        
            if found_in_set and not found_in_interior_set:    
                if(index != target_vtx):
                    print('found',(edge[0],index),(edge[1],index),"Canceling creation")
                    continue        
        
        
        
        
            # Passed edges checking 
            # Proceed to check vertices
            temp_element=(edge[0],edge[1],target_vtx)
            print(temp_element)
            existing_element=False
            for element in set_elements:
                if set(temp_element)== set(element):
                    print("Element {} already in set".format(element))
                    existing_element=True
                    break
            if existing_element:
                break
            
            
            
            if target_vtx in set_locked_vertices:
                print(" Target vertex {} is locked".format(target_vtx))
                continue
            set_elements.add(temp_element)

        
    
          
            
        
        
            # Locking the vertices and checking if the connection is with a locked vertex has been checked/
            # Proceeding to check if both internal edges intersect with other internal edges
            internal_edge1=(edge[0],target_vtx)
            internal_edge2=(edge[1],target_vtx)
            set_a,set_b=get_intermediate_indices(target_vtx,polygon,edge[0],edge[1])
        
            internal_condition1= internal_edge1 in set_forbidden_intersections or tuple(reversed(internal_edge1)) in set_forbidden_intersections
                                                                        
            internal_condition2=internal_edge2 in set_forbidden_intersections or tuple(reversed(internal_edge2)) in set_forbidden_intersections
                                                                            
    
                                                                                   
            internal_intersection=False
            if internal_condition1 or  internal_condition2:
                print("edges :",internal_edge1, "and",internal_edge2,"intersecting")
                print("Abandoning creation of element",temp_element)
                internal_intersection=True
        
     
            if internal_intersection:
                for vtx in temp_element:
                    if Found_locked_vertex and vtx in set_locked_vertices:
                        print("Unlocking vertex",vtx)
                        set_locked_vertices.remove(vtx)                    
                continue
        
        
        
            # Create the element
            element=temp_element
        
            # Add to set of edges all the forbidden intersections after the creation of the element
            
            for i in set_a:
                for j in set_b:
                    set_forbidden_intersections.add((i,j))
            #print("set of forbidden inter section edges updated:",set_forbidden_intersections)
            
            # Check if a locked vertex was created after the creation of the element
            # If so, add it to the list
            #Tracer()()
            Found_locked_vertex=False
            for vertex in element:
                _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
                if isclosed and vertex not in set_locked_vertices:
                    print("Vertex locked:",vertex)
                    Found_locked_vertex=True
                    set_locked_vertices.add(vertex)
                
        
        
        # New edges after creation of the element
   
            new_edge1=(edge[0],target_vtx)
            new_edge2=(edge[1],target_vtx)
        
            if new_edge1 not in set_edges and tuple(reversed(new_edge1)) not in set_edges:
                set_edges.add(new_edge1)
                interior_edges.add(new_edge1)
                print("edges inserted:",new_edge1)
                print("set of interior edges updated:",interior_edges)
                print("set of edges updated:",set_edges)
            if new_edge2 not in set_edges and tuple(reversed(new_edge2)) not in set_edges:    
                set_edges.add(new_edge2)
                interior_edges.add(new_edge2)
                print("edges inserted:",new_edge2)
                print("set of interior edges updated:",interior_edges)
                print("set of edges updated:",set_edges)
            
        
    
    
            # Checking list of elements to see whether the were created or were already there
            
            
            set_elements.add(element)
            element_created=True
                
            if element_created:
                print("element inserted:",element)

                break
            else:
                continue
        
        
            
    
    
    triangulated={'segment_markers': np.ones([polygon.shape[0]]), 'segments':np.array(get_contour_edges(polygon)), 'triangles': np.array(list( list(i) for i in set_elements)),
                  'vertex_markers': np.ones([polygon.shape[0]]), 'vertices': polygon}
    plot.plot(plt.axes(), **triangulated)
    print("Final edges:",set_edges)
    print("Elements created:",set_elements)
    print("Set of locked vertices:", set_locked_vertices)
    
    
    # find open vertices
    for element in set_elements:
        for vertex in  element:
                    _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
                    if isclosed and vertex not in set_locked_vertices:
                        print("Vertex locked:",vertex)
                        Found_locked_vertex=True
                        set_locked_vertices.add(vertex)
    set_open_vertices=set(range(len(polygon)))-set_locked_vertices
    print("Set of open vertices:", set_open_vertices)
    set_edges.clear(),set_locked_vertices.clear(),set_forbidden_intersections.clear
    sub_element_list=[]
    if recursive:     
        sub_polygon_list=check_for_sub_polygon(set_open_vertices,interior_edges,set_elements,polygon)
        for sub_polygon_indices in sub_polygon_list:
            if len(sub_polygon_indices)>=3:
                print("remeshing subpolygon",sub_polygon_indices)
                polygon_copy=polygon
                sub_polygon=np.array(polygon_copy[sub_polygon_indices])
                if not is_counterclockwise(sub_polygon):
                    sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])

                sub_quality,_=quality_matrix(sub_polygon,compute_minimum=True,normalize=False)
                sub_order_matrix=order_quality_matrix(sub_quality,sub_polygon,check_for_equal=True)
                print(sub_quality,sub_order_matrix)
                sub_elements,_=triangulate(sub_polygon,sub_order_matrix,recursive=True)
                if len(sub_elements)!=0:
                    for element in sub_elements:
                        indices=np.asarray(element)
                        print(element)
                        triangle=sub_polygon[indices]
                        polygon_indices=get_indices(triangle,polygon)
                        sub_element_list.append(polygon_indices)
                          
        print(set_elements)
        return set_elements,sub_element_list

    
    
def pure_triangulate(polygon,ordered_quality_matrix,recursive=True):
    set_edges=set(tuple(i) for i in get_contour_edges(polygon))
    interior_edges=set()
    set_elements=set()
    set_locked_vertices=set()
    set_forbidden_intersections=set()
    
    #print("initial set edges:", set_edges)
    


    for edge in ordered_quality_matrix.keys():
        
        found_in_interior_set,found_in_set,index=check_edge_validity(edge,polygon,set_edges,interior_edges)
        
        for qualities_with_edges in ordered_quality_matrix[edge][0]:
            
            element_created=False
           
            target_vtx=qualities_with_edges[1]
            
            if target_vtx==edge[0] or target_vtx==edge[1]:
                continue
           
            #print("Edge:",edge,"targeting:",target_vtx)
        
            if found_in_interior_set:
                element=(edge[0],edge[1],index)  
                set_elements.add(element)
               # print("Element inserted:",element)
                continue
        
            if found_in_set and not found_in_interior_set:    
                if(index != target_vtx):
                   # print('found',(edge[0],index),(edge[1],index),"Canceling creation")
                    continue        
        
        
        
        
            # Passed edges checking 
            # Proceed to check vertices
            temp_element=(edge[0],edge[1],target_vtx)
           # print(temp_element)
            existing_element=False
            for element in set_elements:
                if set(temp_element)== set(element):
                    #print("Element {} already in set".format(element))
                    existing_element=True
                    break
            if existing_element:
                break
            
            
            
            if target_vtx in set_locked_vertices:
                #print(" Target vertex {} is locked".format(target_vtx))
                continue
            set_elements.add(temp_element)

        
    
          
            
        
        
            # Locking the vertices and checking if the connection is with a locked vertex has been checked/
            # Proceeding to check if both internal edges intersect with other internal edges
            internal_edge1=(edge[0],target_vtx)
            internal_edge2=(edge[1],target_vtx)
            set_a,set_b=get_intermediate_indices(target_vtx,polygon,edge[0],edge[1])
        
            internal_condition1= internal_edge1 in set_forbidden_intersections or tuple(reversed(internal_edge1)) in set_forbidden_intersections
                                                                        
            internal_condition2=internal_edge2 in set_forbidden_intersections or tuple(reversed(internal_edge2)) in set_forbidden_intersections
                                                                            
    
                                                                                   
            internal_intersection=False
            if internal_condition1 or  internal_condition2:
                #print("edges :",internal_edge1, "and",internal_edge2,"intersecting")
                #print("Abandoning creation of element",temp_element)
                internal_intersection=True
        
     
            if internal_intersection:
                for vtx in temp_element:
                    if Found_locked_vertex and vtx in set_locked_vertices:
                        #print("Unlocking vertex",vtx)
                        set_locked_vertices.remove(vtx)                    
                continue
        
        
        
            # Create the element
            element=temp_element
        
            # Add to set of edges all the forbidden intersections after the creation of the element
            
            for i in set_a:
                for j in set_b:
                    set_forbidden_intersections.add((i,j))
            #print("set of forbidden inter section edges updated:",set_forbidden_intersections)
            
            # Check if a locked vertex was created after the creation of the element
            # If so, add it to the list
            #Tracer()()
            Found_locked_vertex=False
            for vertex in element:
                _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
                if isclosed and vertex not in set_locked_vertices:
                    #print("Vertex locked:",vertex)
                    Found_locked_vertex=True
                    set_locked_vertices.add(vertex)
                
        
        
        # New edges after creation of the element
   
            new_edge1=(edge[0],target_vtx)
            new_edge2=(edge[1],target_vtx)
        
            if new_edge1 not in set_edges and tuple(reversed(new_edge1)) not in set_edges:
                set_edges.add(new_edge1)
                interior_edges.add(new_edge1)
                #print("edges inserted:",new_edge1)
                #print("set of interior edges updated:",interior_edges)
                #print("set of edges updated:",set_edges)
            if new_edge2 not in set_edges and tuple(reversed(new_edge2)) not in set_edges:    
                set_edges.add(new_edge2)
                interior_edges.add(new_edge2)
                #print("edges inserted:",new_edge2)
                #print("set of interior edges updated:",interior_edges)
                #print("set of edges updated:",set_edges)
            
        
    
    
            # Checking list of elements to see whether the were created or were already there
            
            
            set_elements.add(element)
            element_created=True
                
            if element_created:
                #print("element inserted:",element)

                break
            else:
                continue
        
        
            
    
    
    #triangulated={'segment_markers': np.ones([polygon.shape[0]]), 'segments':np.array(get_contour_edges(polygon)), 'triangles': np.array(list( list(i) for i in set_elements)),
   #               'vertex_markers': np.ones([polygon.shape[0]]), 'vertices': polygon}
    #plot.plot(plt.axes(), **triangulated)
    #print("Final edges:",set_edges)
    #print("Elements created:",set_elements)
    #print("Set of locked vertices:", set_locked_vertices)
    
    
    # find open vertices
    for element in set_elements:
        for vertex in  element:
                    _ ,isclosed = is_closed_ring(vertex,set_elements,*connection_indices(vertex,get_contour_edges(polygon)))
                    if isclosed and vertex not in set_locked_vertices:
                        #print("Vertex locked:",vertex)
                        Found_locked_vertex=True
                        set_locked_vertices.add(vertex)
    set_open_vertices=set(range(len(polygon)))-set_locked_vertices
    #print("Set of open vertices:", set_open_vertices)
    set_edges.clear(),set_locked_vertices.clear(),set_forbidden_intersections.clear
    sub_element_list=[]
    if recursive:     
        sub_polygon_list=pure_check_for_sub_polygon(set_open_vertices,interior_edges,set_elements,polygon)
        for sub_polygon_indices in sub_polygon_list:
            if len(sub_polygon_indices)>=3:
                #print("remeshing subpolygon",sub_polygon_indices)
                polygon_copy=polygon
                sub_polygon=np.array(polygon_copy[sub_polygon_indices])
                if not is_counterclockwise(sub_polygon):
                    sub_polygon=np.array(polygon_copy[sub_polygon_indices[::-1]])

                sub_quality,_=quality_matrix(sub_polygon,compute_minimum=True,normalize=False)
                sub_order_matrix=order_quality_matrix(sub_quality,sub_polygon,check_for_equal=False)
                #print(sub_quality,sub_order_matrix)
                sub_elements,_=pure_triangulate(sub_polygon,sub_order_matrix,recursive=True)
                if len(sub_elements)!=0:
                    for element in sub_elements:
                        indices=np.asarray(element)
                        #print(element)
                        triangle=sub_polygon[indices]
                        polygon_indices=get_indices(triangle,polygon)
                        sub_element_list.append(polygon_indices)
                          
        #print(set_elements)
        return set_elements,sub_element_list
    
    
    
    
def concat_element_list(set_elements,sub_element_list):
    list_set_elements=[list(i) for i in set_elements]
    return list_set_elements+sub_element_list


    
def order_quality_matrix(_quality_matrix,_polygon, check_for_equal=True):

    #  Create the quality matrix in accordance with the edges
    quality_board=[(q,index)  for qualities in _quality_matrix for index,q in enumerate(qualities)]
    quality_board=np.array(quality_board)
    #print("Quality board not resized:",quality_board)

    quality_board.resize(len(get_contour_edges(_polygon)),len(_polygon),2)
    quality_board=dict(zip(list(tuple(i) for i in get_contour_edges(_polygon)),quality_board))
    
    #sorted_quality_board={i[0]:i[1] for i in sorted(board.items(),key=lambda x: max(x[1]),reverse=True)}
    #print("Quality board")
    #for keys,items in quality_board.items():
    #    print(keys,items)
    edge_quality=quality_board[(0,1)]
    edge_quality=edge_quality[np.lexsort(np.fliplr(edge_quality).T)]




    for i in quality_board.keys():
        quality_board[i]=quality_board[i][np.lexsort(np.fliplr(quality_board[i]).T)]
        quality_board[i]=quality_board[i][::-1]
        quality_board[i][:,1]=quality_board[i][:,1].astype(int)
    
    listing=[]
    for keys,values in quality_board.items():
        listing.append([keys,max(values[:,0])])
    
    listing=np.array(listing)
    listing=listing[np.lexsort(np.transpose(listing)[::-3]).T]
    listing=listing[::-1]
    ordered_indices=listing[:,0]

    ordered_quality_matrix={}
    
    for i in ordered_indices:
        ordered_quality_matrix[i]=[tuple(zip(quality_board[i][:,0],quality_board[i][:,1].astype(int)))]
    
    if check_for_equal:
        ordered_quality_matrix=check_ordered_matrix(ordered_quality_matrix,_polygon)
    return ordered_quality_matrix
        
    
    
    

def check_ordered_matrix(_order_matrix,polygon):
    
    checked_matrix=copy.deepcopy(_order_matrix)
    listing=np.empty([len(checked_matrix),len(checked_matrix)],dtype=np.float32)
    for i,keys in enumerate(checked_matrix):
        
        for qualities_with_indices in  checked_matrix[keys]:
            
            for j,(qualities,indices) in enumerate(qualities_with_indices):
                
                #print(qualities,indices,'\n')
                listing[i,j]=qualities
    edge_list=list(checked_matrix.keys())
   # listing=listing[::-1]
    for ind,i in enumerate(listing):
       # print("checking",edge_list[ind])
        non_zero_list=i[np.where(i!=0)]
        #non_zero_list=non_zero_list[::-1]
        unique_non_zero_list,count=np.unique(non_zero_list,return_counts=True)
        unique_non_zero_list=unique_non_zero_list[::-1]
        count=count[::-1]
        value_with_counts=list(zip(unique_non_zero_list,count))

        for j in value_with_counts:
            lst=list(checked_matrix[edge_list[ind]][0])

            if j[1]>1:
#                pdb.set_trace()
                indices=[]
                
                connection_vertex_with_mean_qualities=[]
                tag=j[0]
                for index,j in enumerate(non_zero_list):
                    if tag==j:
                        print(index)
                        indices.append(index)
                        connection_vertex=int(checked_matrix[edge_list[ind]][0][index][1])
                        triangle_indices=np.asarray([edge_list[ind][0],edge_list[ind][1],connection_vertex])                
                        print("triangle",triangle_indices)
                        triangle=polygon[triangle_indices]
                        mean_quality=compute_mean_quality_triangle(triangle,polygon)
                        print(mean_quality)
                        connection_vertex_with_mean_qualities.append(tuple((mean_quality,connection_vertex)))
                connection_vertex_with_mean_qualities=np.array(connection_vertex_with_mean_qualities,dtype='float32,uint16')
               # connection_vertex_with_mean_qualities[:,1]= connection_vertex_with_mean_qualities[:,1].astype(int)
                print(connection_vertex_with_mean_qualities)
                sorted_array=np.sort(connection_vertex_with_mean_qualities,axis=0)
                sorted_array=sorted_array[::-1]

                sorted_array=[tuple(i) for i in sorted_array]
                print(sorted_array)
                for index,k in enumerate(indices): 
                    lst[k]=sorted_array[index]
            #print("replacing {} \n with {}:".format(checked_matrix[edge_list[ind]][0],lst))
            checked_matrix[edge_list[ind]][0]=tuple(lst)
        print("checked",edge_list[ind])
                
                    
                    
    
    
    return checked_matrix
    
    
    
    



# Function to get the list of edges of a polygon
def get_contour_edges(polygon):
    contour_connectivity=np.array([[i,(i+1)%polygon.shape[0]] for i in range(polygon.shape[0])])
    return contour_connectivity



# Function to return indices that are connected to a vertex
def connection_indices(vertex,edges):   
    indices=[]
    for edge in edges:
        if vertex in edge:

            if edge[0] == vertex:
                indices.append(edge[1])
            else:
                indices.append(edge[0])

    return indices

# Function to calculate and angle:
def calculate_angle(p0,p1,p2):
    v0 = p1 - p0
    v1 = p2 - p0
    
    
    #normal=compute_triangle_normals([p0,p1,p2])
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    angle=abs(angle)
    #unit_v0=v0 / np.linalg.norm(v0)
    #unit_v1=v1 / np.linalg.norm(v1)
    #angle=np.arccos(np.clip(np.dot(unit_v0, unit_v1), -1.0, 1.0))
    
    return np.degrees(angle)



# Function to calculate the angles of a polygon
def get_polygon_angles(polygon):
    angles=[]
    for index,point in enumerate(polygon):
        p0=point        
        neighbor_points=connection_indices(index,get_contour_edges(polygon))
        #print("neighbor points",neighbor_points)
        indices=np.asarray(neighbor_points)
        p1,p2=polygon[indices]
        angle=calculate_angle(p0,p1,p2)
        if index !=0:
            triangle_normal=compute_triangle_normals([p0,p1,p2])
        else:
            triangle_normal=compute_triangle_normals([p1,p0,p2])

            
        if triangle_normal>0:
            angle=360-angle
        
        angles.append(angle)
    return angles
    



def is_closed_ring(vtx,set_of_elements,*adj_vtx):
    contour_edge1=(vtx,adj_vtx[0])
    contour_edge2=(vtx,adj_vtx[1])
    visited_elements=set_of_elements.copy()
    
    target_edge=contour_edge1
    
    edges_found=[]
    edges_found.append(contour_edge1)

    proceed=True
    
    while proceed:
        
        if not visited_elements:
            break
        
        remaining_edge,found_element=edge2elem(target_edge,visited_elements)
        
        if found_element is None:
            #print("stopped")
            proceed=False
            break
            
        visited_elements.remove(found_element)
        edges_found.append(remaining_edge)
        target_edge=remaining_edge
    
    
                
    #print(set(edges_found))
    found_contour_edge1,found_contour_edge2=False,False
    found_contour_edges=False

    # Checking if both contour edges area contained in the set of edges acquired
    
    for edge in edges_found:
        condition1= contour_edge1[0] in set(edge) and contour_edge1[1] in set((edge))
        condition2= contour_edge2[0] in set(edge) and contour_edge2[1] in set((edge))
        if condition1:
            #print("found ",contour_edge1)
            found_contour_edge1=True
        if condition2:
            #print("found",contour_edge2)
            found_contour_edge2=True
            
    if found_contour_edge1 and found_contour_edge2:
        found_contour_edges=True
        #print("found both of contour edges in set")
    
    visited_elements.clear()
    return edges_found,found_contour_edges
    
    
    
# Finds element containing the edge and exits (does not give the full list of elements)   
# Serve is_one_ring function
def edge2elem(edge,set_of_elements):
    Found_element=()
    Remaining_edge=()
  
    for element in set_of_elements.copy():
        
        if edge[0] in  set(element) and edge[1] in element: 
            #print("Edge {} is part of element {}".format(edge,element))
            Found_element=element
            Remaining_index=set(element)-set(edge)
            Remaining_index=list(Remaining_index)
            Remaining_edge=(edge[0],Remaining_index[0])
            #print(" Remaining edge is {}".format(Remaining_edge))
            break 
        else:
            Found_element=None
            Remaining_edge=None
    return  Remaining_edge,Found_element 




# Departing from a target vertex connected with and edge get all intermediate  indices from one side and other
def get_intermediate_indices(target_vtx,polygon,*edge):
    
    set_1=set()
    set_2=set()
    
    
    contour_edges=get_contour_edges(polygon)
    
    
    # Depart from target vertex and get neighbor indices
    neighbors=connection_indices(target_vtx,contour_edges)
    found_vertex1,found_vertex2=neighbors[0],neighbors[1]
    #print("found vertices:",found_vertex1,found_vertex2)

    
    # Include them into seperate lists
    set_1.add(found_vertex1)
    set_2.add(found_vertex2)
    
    visited_vertex=target_vtx
      
    
    while found_vertex1!=edge[0] and found_vertex1!=edge[1]:
        visiting_vertex=found_vertex1
        neighbors=connection_indices(visiting_vertex,contour_edges)
        for index in neighbors:
            if index !=  visited_vertex:
                set_1.add(index)
                found_vertex1=index
                #print("Found vertex:",found_vertex1)     
        visited_vertex=visiting_vertex
        
    #print("Start  looking the other way")
    
    # Resetting to go the other way
    visited_vertex=target_vtx

    while found_vertex2!=edge[0] and found_vertex2!=edge[1]:
        visiting_vertex=found_vertex2
        neighbors=connection_indices(visiting_vertex,contour_edges)
        for index in neighbors:
            if index !=  visited_vertex:
                set_2.add(index)
                found_vertex2=index
                #print("Found vertex:",found_vertex2)     
        visited_vertex=visiting_vertex
    
                
                
                
  
    return set_1,set_2
                
            
        
    
    


# In[8]:

def polygon_2_vtx(starting_vertex,edges_to_visit,set_of_common_vertices,pair_of_adjacent_edges):
    from  more_itertools import unique_everseen
    
    if not edges_to_visit:
        return
    
    
    closed=False
    
    #pdb.set_trace()
    
    print("Edges to visit:",edges_to_visit)
    subpolygon=[]
    
    set_of_points=set([j for i in edges_to_visit for j in i])
    
    if starting_vertex not in set_of_points:
        return    
                
    found_vertex=starting_vertex
    target_edge=[]

    while not closed:
        for index,edge in enumerate(edges_to_visit.copy()):
            visiting_vertex=found_vertex
            
            
            if target_edge:
 #               pdb.set_trace()
                if edge!= target_edge[0] and  edge!= tuple(reversed(target_edge[0])) :
                    continue           
                else:
                    target_edge.pop()
            #if visiting_vertex not in set(edge) and index==len(edges_to_visit.copy()):
               # Tracer()()
                #print("Not found in list of edges")
                #closed=True
                #break
            if visiting_vertex not in set(edge):
                continue
            subpolygon.append(visiting_vertex)
                
                                
            print("Visiting vertex",visiting_vertex)
            
        #    found_starting_vtx=False
            subpolygon.append(found_vertex)
            
            
            print(visiting_vertex," in ", edge)
                
           

          
                
            for index in set(edge):
                if visiting_vertex!= index:
                    found_vertex=index
                    print("Found vertex:",found_vertex)
                    subpolygon.append(found_vertex)
            found_crossroad=False
            found_in_set=False
            # Check if edge is part of a crossroad (check if found vertex is point of multiple polygons)
            if found_vertex in set_of_common_vertices:
                found_crossroad=True
          
            # If yes then the next visiting edge should be the one is the pair of adjacent edges
            if found_crossroad:
                for  edges_in_same_polygon in pair_of_adjacent_edges.copy(): 
                    if edge in set(edges_in_same_polygon) or tuple(reversed(edge)) in set(edges_in_same_polygon):
                        for edges in edges_in_same_polygon:
                            if edges!=edge and edges!=tuple(reversed(edge)):
                                target_edge.append(edges)
                                found_in_set=True
                                print("edge {} should be followed by {}".format(edge,edges))
                                pair_of_adjacent_edges.discard(edges_in_same_polygon)
                                
                                break
                    if found_in_set:
                        break
                    
            print("Removing edge",edge)
            edges_to_visit.discard(edge)
            print(edges_to_visit)
            if found_vertex==starting_vertex:
                subpolygon=list(unique_everseen(subpolygon))
                print("Back to starting vertex")    
                closed=True
                break
                
    if  len(subpolygon)<3:
        return 
    else:
        return subpolygon
     

def pure_polygon_2_vtx(starting_vertex,edges_to_visit,set_of_common_vertices,pair_of_adjacent_edges):
    from  more_itertools import unique_everseen
    
    if not edges_to_visit:
        return
    
    
    closed=False
    
    #pdb.set_trace()
    
    #print("Edges to visit:",edges_to_visit)
    subpolygon=[]
    
    set_of_points=set([j for i in edges_to_visit for j in i])
    
    if starting_vertex not in set_of_points:
        return    
                
    found_vertex=starting_vertex
    target_edge=[]

    while not closed:
        for index,edge in enumerate(edges_to_visit.copy()):
            visiting_vertex=found_vertex
            
            
            if target_edge:
 #               pdb.set_trace()
                if edge!= target_edge[0] and  edge!= tuple(reversed(target_edge[0])) :
                    continue           
                else:
                    target_edge.pop()
            #if visiting_vertex not in set(edge) and index==len(edges_to_visit.copy()):
               # Tracer()()
                #print("Not found in list of edges")
                #closed=True
                #break
            if visiting_vertex not in set(edge):
                continue
            subpolygon.append(visiting_vertex)
                
                                
            #print("Visiting vertex",visiting_vertex)
            
        #    found_starting_vtx=False
            subpolygon.append(found_vertex)
            
            
            #print(visiting_vertex," in ", edge)
                
           

          
                
            for index in set(edge):
                if visiting_vertex!= index:
                    found_vertex=index
                    #print("Found vertex:",found_vertex)
                    subpolygon.append(found_vertex)
            found_crossroad=False
            found_in_set=False
            # Check if edge is part of a crossroad (check if found vertex is point of multiple polygons)
            if found_vertex in set_of_common_vertices:
                found_crossroad=True
          
            # If yes then the next visiting edge should be the one is the pair of adjacent edges
            if found_crossroad:
                for  edges_in_same_polygon in pair_of_adjacent_edges.copy(): 
                    if edge in set(edges_in_same_polygon) or tuple(reversed(edge)) in set(edges_in_same_polygon):
                        for edges in edges_in_same_polygon:
                            if edges!=edge and edges!=tuple(reversed(edge)):
                                target_edge.append(edges)
                                found_in_set=True
                                #print("edge {} should be followed by {}".format(edge,edges))
                                pair_of_adjacent_edges.discard(edges_in_same_polygon)
                                
                                break
                    if found_in_set:
                        break
                    
            #print("Removing edge",edge)
            edges_to_visit.discard(edge)
            #print(edges_to_visit)
            if found_vertex==starting_vertex:
                subpolygon=list(unique_everseen(subpolygon))
                #print("Back to starting vertex")    
                closed=True
                break
                
    if  len(subpolygon)<3:
        return 
    else:
        return subpolygon
                            

def pure_check_for_sub_polygon(set_of_open_vertices,set_of_interior_edges,set_of_elements,polygon):

    set_polygon_edges=set(tuple(i) for i in get_contour_edges(polygon))

    
    if not set_of_open_vertices or  len(set_of_open_vertices)<3:
        return []
    

    sub_polygon_list=[]
    modified_interior_edge_set=set_of_interior_edges.copy()
    
 
    
    
    polygon_connectivity=[tuple(i) for i in get_contour_edges(polygon)]
    
    for edge in modified_interior_edge_set.copy():
        if edge[0] not in set_of_open_vertices or edge[1] not in set_of_open_vertices:
            modified_interior_edge_set.discard(edge)
   



    # Taking care of vertices that are locked but the element is not seen
    
    set_of_unfound_locked_vertices=set()
    continue_looking=True

    
    while continue_looking:
        
        if not set_of_open_vertices:
            continue_looking=False
            
        for vtx in set_of_open_vertices.copy():
                vtx1,vtx2 =connection_indices(vtx,get_contour_edges(polygon))
                found_edges1,isclosed1=is_closed_ring(vtx,set_of_elements,vtx2,vtx1)
                found_edges2,isclosed2=is_closed_ring(vtx,set_of_elements,vtx1,vtx2)
                #print("Examining if vtx {} is locked".format(vtx))
                
                if isclosed1 or isclosed2:
                    #print(vtx,"locked after all")
                    set_of_open_vertices.discard(vtx)
                    for edge in modified_interior_edge_set.copy():
                        if vtx in edge:
                            modified_interior_edge_set.discard(edge)
                    break
                
                for edge in found_edges1:
                    if edge in polygon_connectivity or edge[::-1] in polygon_connectivity:
                        found_edges1.remove(edge)
                for edge in found_edges2:
                    if edge in polygon_connectivity or edge[::-1] in polygon_connectivity:
                        found_edges2.remove(edge)
                between_edges=[]
                for edge in found_edges1:
                    for indices in edge:
                        if indices==vtx:
                            continue
                    between_edges.append(indices)
                for edge in found_edges2:
                    for indices in edge:
                        if indices==vtx:
                            continue
                    between_edges.append(indices)
                for edge in set_of_interior_edges.copy():
                    found_locked_vtx=False
                    if set(between_edges)==set(edge):
                        #print(vtx,"locked after all")
                        found_locked_vtx=True
                        set_of_unfound_locked_vertices.add(vtx)
                        #Tracer()()
                        if edge in set_of_interior_edges or edge[::-1] in set_of_interior_edges:                 
                            #modified_interior_edge_set.discard(edge)
                            #print(edge,"removed")               
                            #modified_interior_edge_set.discard(edge[::-1])
                            modified_interior_edge_set.discard((vtx,between_edges[0]))
                            modified_interior_edge_set.discard((between_edges[0],vtx))
                        
    
                            modified_interior_edge_set.discard((vtx,between_edges[1]))
                            modified_interior_edge_set.discard((between_edges[1],vtx))
                            element=(vtx,between_edges[0],between_edges[1])
                            #print("Removed:",(vtx),"from set of open vertices")
    
                            #print("Added new element:",element)
                            #print("Removed:",(vtx,between_edges[0]),"from set of edges")
                            #print("Removed:",(vtx,between_edges[1]),"from set of edges")
    
                            set_of_elements.add(element) 
                            #print("New set of elements",set_of_elements)
                            set_of_open_vertices.discard(vtx)
                            
                    if found_locked_vtx:
                        #Tracer()()
                        continue_looking=True
                        #print("Re-evaluting set of open vertices")
                        break
                        
                    else: continue_looking=False
                        
                        
    #    for edge in modified_interior_edge_set.copy():
    #        if set(edge).issubset(set_of_unfound_locked_vertices):
    #            modified_interior_edge_set.discard(edge)
#            modified_interior_edge_set.discard(edge[::-1])
#            print("removed",edge)
            
            #print("inbetween",between_edges)
                
    #print("set of open vertices",set_of_open_vertices)
    
    if not set_of_open_vertices or  len(set_of_open_vertices)<3:
        return []
    
    # In the set of open vertices there may be vertices that are part of  of multiple polygon
    #found_common_vertex=False
   
    set_of_common_vertices=set()
    pair_of_adjacent_edges=set()
    for vertex in set_of_open_vertices:
        nb_of_polygon=0
        count=0
        for edge in modified_interior_edge_set.copy():
            counter2=0
            if vertex in set(edge):
                count+=1
            for element in set_of_elements:
                if set(edge).issubset(set(element)):
                    counter2+=1
            if counter2==2:
                #print("Edge {} is common for two elemenets".format(edge))
                count-=1
                modified_interior_edge_set.discard(edge)
                
        if count>=3:
            adj_vertices=sorted(list(vtx for edge in modified_interior_edge_set if vertex  in set(edge) for  vtx in edge if vtx!=vertex))
            #print("Vertex {} surrounded by {}".format(vertex,adj_vertices))
            counter=0
            # Checking if vertice are linked , if they are then aren't part of the same polygon
            for index,_ in enumerate(adj_vertices):
                edge=tuple((adj_vertices[index],adj_vertices[(index+1)%len(adj_vertices)]))
                condition3=True
                # CHECK  CONDITION TO FIND OUT WHICH EDGE IS PAIRED WITH WHICH TO FORM A POLYGON
                
                
                # Connections could form elements that are not discovered
                if ((edge in set_of_interior_edges or tuple(reversed(edge)) in set_of_interior_edges )and
                ((vertex,edge[0]) in set_of_interior_edges or tuple(reversed((vertex,edge[0]))) in set_of_interior_edges) and
                ((vertex,edge[1]) in set_of_interior_edges or tuple(reversed((vertex,edge[1]))) in set_of_interior_edges) ):
                    #print("Found new element:",(vertex,edge[0],edge[1]))
                    #print("({},{}) and ({},{}) are part of the same element".format(edge[0],vertex,edge[1],vertex))
                    pair_of_adjacent_edges.add((((edge[0],vertex),(edge[1],vertex))))
                    continue
                
                if(edge[0]<edge[1]):
                    for i in range(edge[0]+1,edge[1]) :
                        if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges 
                            or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                            if i==vertex:continue
                            condition3=False
                                       
                else:
                    for i in range(edge[0]+1,len(polygon)-1) :
                        if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges 
                            or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                            if i==vertex:continue
                            condition3=False
                    for i in range(edge[1]) :
                        if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges 
                            or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                            if i == vertex:continue
                            condition3=False
                
                condition1=edge  in set_polygon_edges 
                condition2=tuple(reversed(edge)) in set_polygon_edges
                nb_of_polygons=[]
                if not condition1 and not condition2  and condition3 :
                    counter+=1
                    #print("({},{}) and ({},{}) are part of the same polygon".format(edge[0],vertex,edge[1],vertex))
                    pair_of_adjacent_edges.add((((edge[0],vertex),(edge[1],vertex))))
            set_of_common_vertices.add(vertex)
            nb_of_polygons.append(counter)
            #print("vertex {} is adjacent to {} polygons".format(vertex,counter))
            #print("Set of adjacent edges to visit:",pair_of_adjacent_edges)
       # found_common_vertex=True
        
    # An edge could be part of more than one polygons. This means that the vertices of this edge
    # are already in the set of common vertices and the edges is inside the set of the of modi
    # fied interior edges
   # set_of_common_edges=set()
    #for vtx1 in  set_of_common_vertices:
     #   for vtx2 in set_of_common_vertices:
      #      pass
    
    
    
    
    
    # if the set found is les than 4 then now polygon is formed
    if len(set_of_open_vertices)<4:
        return []
    
    edges_to_visit=modified_interior_edge_set

    
    sub_polygon_list=[]
 #   pdb.set_trace()

    try:
        if set_of_common_vertices:
            for vtx in set_of_common_vertices:
                subpolygon=pure_polygon_2_vtx(vtx,edges_to_visit,set_of_common_vertices,pair_of_adjacent_edges)
                if subpolygon is not None:
                   sub_polygon_list.append(subpolygon)
        #print(sub_polygon_list)
    except:
        #print("Failed")
        pass
    
    
    # Removing eges where one vertex is locked and the other is not:
    for edge in edges_to_visit.copy():
        if (edge[0] in set_of_open_vertices and edge[1] not in set_of_open_vertices) or (edge[1] in set_of_open_vertices and edge[0] not in set_of_open_vertices):
            edges_to_visit.discard(edge)
            #print("Removing",edge,"from edges to visit")
            #print("Edges to visit are now",edges_to_visit)
    
    
    while edges_to_visit:          
        for vtx in set_of_open_vertices.copy():
            #print("Starting with vertex",vtx)      
            subpolygon=pure_polygon_2_vtx(vtx,edges_to_visit,set_of_common_vertices,pair_of_adjacent_edges)

            if subpolygon is not None:
                sub_polygon_list.append(subpolygon)
                #print(sub_polygon_list)
        
                                    
#    for sub_polygon in sub_polygon_list:
#        if len(sub_polygon)>3:
#            print("found polygon",sub_polygon)
#        else:
#            print("found element",sub_polygon)
    return sub_polygon_list

                           
def check_for_sub_polygon(set_of_open_vertices,set_of_interior_edges,set_of_elements,polygon):

    set_polygon_edges=set(tuple(i) for i in get_contour_edges(polygon))

    
    if not set_of_open_vertices or  len(set_of_open_vertices)<3:
        return []
    

    sub_polygon_list=[]
    modified_interior_edge_set=set_of_interior_edges.copy()
    
 
    
    
    polygon_connectivity=[tuple(i) for i in get_contour_edges(polygon)]
    
    for edge in modified_interior_edge_set.copy():
        if edge[0] not in set_of_open_vertices or edge[1] not in set_of_open_vertices:
            modified_interior_edge_set.discard(edge)
   



    # Taking care of vertices that are locked but the element is not seen
    
    set_of_unfound_locked_vertices=set()
    continue_looking=True

    
    while continue_looking:
        
        if not set_of_open_vertices:
            continue_looking=False
            
        for vtx in set_of_open_vertices.copy():
                vtx1,vtx2 =connection_indices(vtx,get_contour_edges(polygon))
                found_edges1,isclosed1=is_closed_ring(vtx,set_of_elements,vtx2,vtx1)
                found_edges2,isclosed2=is_closed_ring(vtx,set_of_elements,vtx1,vtx2)
                print("Examining if vtx {} is locked".format(vtx))
                
                if isclosed1 or isclosed2:
                    print(vtx,"locked after all")
                    set_of_open_vertices.discard(vtx)
                    for edge in modified_interior_edge_set.copy():
                        if vtx in edge:
                            modified_interior_edge_set.discard(edge)
                    break
                
                for edge in found_edges1:
                    if edge in polygon_connectivity or edge[::-1] in polygon_connectivity:
                        found_edges1.remove(edge)
                for edge in found_edges2:
                    if edge in polygon_connectivity or edge[::-1] in polygon_connectivity:
                        found_edges2.remove(edge)
                between_edges=[]
                for edge in found_edges1:
                    for indices in edge:
                        if indices==vtx:
                            continue
                    between_edges.append(indices)
                for edge in found_edges2:
                    for indices in edge:
                        if indices==vtx:
                            continue
                    between_edges.append(indices)
                for edge in set_of_interior_edges.copy():
                    found_locked_vtx=False
                    if set(between_edges)==set(edge):
                        print(vtx,"locked after all")
                        found_locked_vtx=True
                        set_of_unfound_locked_vertices.add(vtx)
                        #Tracer()()
                        if edge in set_of_interior_edges or edge[::-1] in set_of_interior_edges:                 
                            #modified_interior_edge_set.discard(edge)
                            #print(edge,"removed")               
                            #modified_interior_edge_set.discard(edge[::-1])
                            modified_interior_edge_set.discard((vtx,between_edges[0]))
                            modified_interior_edge_set.discard((between_edges[0],vtx))
                        
    
                            modified_interior_edge_set.discard((vtx,between_edges[1]))
                            modified_interior_edge_set.discard((between_edges[1],vtx))
                            element=(vtx,between_edges[0],between_edges[1])
                            print("Removed:",(vtx),"from set of open vertices")
    
                            print("Added new element:",element)
                            print("Removed:",(vtx,between_edges[0]),"from set of edges")
                            print("Removed:",(vtx,between_edges[1]),"from set of edges")
    
                            set_of_elements.add(element) 
                            print("New set of elements",set_of_elements)
                            set_of_open_vertices.discard(vtx)
                            
                    if found_locked_vtx:
                        #Tracer()()
                        continue_looking=True
                        print("Re-evaluting set of open vertices")
                        break
                        
                    else: continue_looking=False
                        
                        
    #    for edge in modified_interior_edge_set.copy():
    #        if set(edge).issubset(set_of_unfound_locked_vertices):
    #            modified_interior_edge_set.discard(edge)
#            modified_interior_edge_set.discard(edge[::-1])
#            print("removed",edge)
            
            #print("inbetween",between_edges)
                
    print("set of open vertices",set_of_open_vertices)
    
    if not set_of_open_vertices or  len(set_of_open_vertices)<3:
        return []
    
    # In the set of open vertices there may be vertices that are part of  of multiple polygon
    #found_common_vertex=False
   
    set_of_common_vertices=set()
    pair_of_adjacent_edges=set()
    for vertex in set_of_open_vertices:
        nb_of_polygon=0
        count=0
        for edge in modified_interior_edge_set.copy():
            counter2=0
            if vertex in set(edge):
                count+=1
            for element in set_of_elements:
                if set(edge).issubset(set(element)):
                    counter2+=1
            if counter2==2:
                print("Edge {} is common for two elemenets".format(edge))
                count-=1
                modified_interior_edge_set.discard(edge)
                
        if count>=3:
            adj_vertices=sorted(list(vtx for edge in modified_interior_edge_set if vertex  in set(edge) for  vtx in edge if vtx!=vertex))
            print("Vertex {} surrounded by {}".format(vertex,adj_vertices))
            counter=0
            # Checking if vertice are linked , if they are then aren't part of the same polygon
            for index,_ in enumerate(adj_vertices):
                edge=tuple((adj_vertices[index],adj_vertices[(index+1)%len(adj_vertices)]))
                condition3=True
                # CHECK  CONDITION TO FIND OUT WHICH EDGE IS PAIRED WITH WHICH TO FORM A POLYGON
                
                
                # Connections could form elements that are not discovered
                if ((edge in set_of_interior_edges or tuple(reversed(edge)) in set_of_interior_edges )and
                ((vertex,edge[0]) in set_of_interior_edges or tuple(reversed((vertex,edge[0]))) in set_of_interior_edges) and
                ((vertex,edge[1]) in set_of_interior_edges or tuple(reversed((vertex,edge[1]))) in set_of_interior_edges) ):
                    print("Found new element:",(vertex,edge[0],edge[1]))
                    print("({},{}) and ({},{}) are part of the same element".format(edge[0],vertex,edge[1],vertex))
                    pair_of_adjacent_edges.add((((edge[0],vertex),(edge[1],vertex))))
                    continue
                
                if(edge[0]<edge[1]):
                    for i in range(edge[0]+1,edge[1]) :
                        if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges 
                            or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                            if i==vertex:continue
                            condition3=False
                                       
                else:
                    for i in range(edge[0]+1,len(polygon)-1) :
                        if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges 
                            or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                            if i==vertex:continue
                            condition3=False
                    for i in range(edge[1]) :
                        if ((vertex,i)  in set_of_interior_edges or (vertex,i)  in set_polygon_edges 
                            or  (i,vertex) in set_of_interior_edges or (i,vertex) in set_polygon_edges):
                            if i == vertex:continue
                            condition3=False
                
                condition1=edge  in set_polygon_edges 
                condition2=tuple(reversed(edge)) in set_polygon_edges
                nb_of_polygons=[]
                if not condition1 and not condition2  and condition3 :
                    counter+=1
                    print("({},{}) and ({},{}) are part of the same polygon".format(edge[0],vertex,edge[1],vertex))
                    pair_of_adjacent_edges.add((((edge[0],vertex),(edge[1],vertex))))
            set_of_common_vertices.add(vertex)
            nb_of_polygons.append(counter)
            print("vertex {} is adjacent to {} polygons".format(vertex,counter))
            print("Set of adjacent edges to visit:",pair_of_adjacent_edges)
       # found_common_vertex=True
        
    # An edge could be part of more than one polygons. This means that the vertices of this edge
    # are already in the set of common vertices and the edges is inside the set of the of modi
    # fied interior edges
   # set_of_common_edges=set()
    #for vtx1 in  set_of_common_vertices:
     #   for vtx2 in set_of_common_vertices:
      #      pass
    
    
    
    
    
    # if the set found is les than 4 then now polygon is formed
    if len(set_of_open_vertices)<4:
        return []
    
    edges_to_visit=modified_interior_edge_set

    
    sub_polygon_list=[]
 #   pdb.set_trace()

    try:
        if set_of_common_vertices:
            for vtx in set_of_common_vertices:
                subpolygon=polygon_2_vtx(vtx,edges_to_visit,set_of_common_vertices,pair_of_adjacent_edges)
                if subpolygon is not None:
                   sub_polygon_list.append(subpolygon)
        print(sub_polygon_list)
    except:
        print("Failed")
    
    
    
    # Removing eges where one vertex is locked and the other is not:
    for edge in edges_to_visit.copy():
        if (edge[0] in set_of_open_vertices and edge[1] not in set_of_open_vertices) or (edge[1] in set_of_open_vertices and edge[0] not in set_of_open_vertices):
            edges_to_visit.discard(edge)
            print("Removing",edge,"from edges to visit")
            print("Edges to visit are now",edges_to_visit)
    
    
    while edges_to_visit:          
        for vtx in set_of_open_vertices.copy():
            print("Starting with vertex",vtx)      
            subpolygon=polygon_2_vtx(vtx,edges_to_visit,set_of_common_vertices,pair_of_adjacent_edges)

            if subpolygon is not None:
                sub_polygon_list.append(subpolygon)
                print(sub_polygon_list)
        
                                    
    for sub_polygon in sub_polygon_list:
        if len(sub_polygon)>3:
            print("found polygon",sub_polygon)
        else:
            print("found element",sub_polygon)
    return sub_polygon_list            
                              
        


def export_contour(filename,contour):
    path=os.path.join('contour_cases',filename+'.txt')
    file=open(path,'w')
    for i in contour:
        file.write(np.array2string(i)+"\n")
    file.close()

    
def read_contour(filename):
    path=os.path.join('contour_cases',filename+'.txt')
    contour=[]
    file=open(path,'r')
    for line in file:
        coord=np.fromstring(line.strip('[\n]'), dtype=float, sep=' ')
        contour.append(coord)
    file.close()
    return np.array(contour)

