import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import numpy
import numpy.linalg
import math
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import re


# Functions f (can be a projection)
#Euclidean distance in dimension n
def d(x,y):
    n=len(x)
    S=0
    for i in range(n):
        S+=(x[i] - y[i])**2
    return math.sqrt(S)

def d_from_x_axis(p):
    return d(p,[p[0],0,0])

def d_from_y_axis(p):
    return d(p,[0,p[1],0])

def d_from_z_axis(p):
    return d(p,[0,0,p[2]])

def proj_x(p):
    return p[0]

def proj_y(p):
    return p[1]

def proj_z(p):
    return p[2]

#Covering U
def covering(m,M,k,p): #m and M are min and max of the data list, k is the number of wanted spans, p percentage of overlapping
    l=(M-m)/k*(1+p)
    overlap=l*p
    cover=[]
    a=m
    b=m+l
    for i in range(k):
        cover.append((a,b))
        a=b-overlap
        b=a+l
    return cover

#return True iff x is in the interval U
def in_interval(x,U):
    return (x>=U[0]) and (x<=U[1])

#return list of f^(-1)(U_i), given the list of images
def pre_image(Y,U,X):
    L=[]
    n=len(Y)
    for u in U:
        L_i=[]
        for i in range(n):
            if in_interval(Y[i],u):
                L_i.append(X[i])
        L.append(L_i)
    return L

def intersection(A,B):
    L=[]
    for a in A:
        if a in B:
            L.append(a)
    return L

def graph(clusters):
    k=len(clusters)
    G=[[i] for i in range(k)]
    for i in range(k-1):
        for j in range(i+1,k):
            if intersection(clusters[i],clusters[j]) != []:
                G.append([i,j])
    return G


#Return the list of points in the ply file
def export(file):
    with open(file, "r") as f:
        line = None
        while line != 'end_header\n':
            line = f.readline()
        
        points = []
        while True:
            line = f.readline()[:-2]
            nbrs = line.split(' ')
            print('line', line)
            print('nbrs', nbrs)
            if len(nbrs) == 3:
                points.append(tuple(float(num) for num in nbrs))
            else:
                break
    return points




def convert(X):
    X = pd.DataFrame(X, columns = ['abscisse','ordonnée'])


def nearest_neighbor(X):
    plt.figure()
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(X)
    distances, indices = nbrs.kneighbors(X)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

def clustering(X,eps):  #Does not give nice results
    #plt.figure()
    clus = DBSCAN(eps, min_samples=5).fit(X)
    return clus.labels_
    #plt.scatter(X['abscisse'],X['ordonnée'],c = y_pred)
    #plt.show()

def K_means(X,k):
    clus=KMeans(n_clusters=k, random_state=0, n_init="auto").fit_predict(X)
    n=len(X)
    C=[[] for i in range(k)]
    for i in range(n):
        j=clus[i]
        C[j].append(X[i])
    return C

def hierarchical(X,k):
    hierarchical_cluster = AgglomerativeClustering(n_clusters=k, metric='euclidean', linkage='ward')
    clus = hierarchical_cluster.fit_predict(X)
    n=len(X)
    C=[[] for i in range(k)]
    for i in range(n):
        j=clus[i]
        C[j].append(X[i])
    return C
    
def barycenter(points):
    dim=len(points[0])
    n=len(points)
    S=[0.]*dim
    for x in points:
        for i in range(dim):
            S[i]+=x[i]
    for j in range(dim):
        S[j]=S[j]/n
    return S

def centers(clusters):
    C=[]
    for clus in clusters:
        C.append(barycenter(clus))
    return C

def display(X):
    fig = plt.figure()
    ax = plt.axes(projection='3d')  # Affichage en 3D
    ax.scatter([x[0] for x in X], [x[1] for x in X], [x[2] for x in X], label='Courbe', marker='d')
    plt.show()




#Computation of Matrix corresponding to the 2-dim simplicial complex
def boundary_matrix(X):
    V=[]
    E=[]
    for scx in X:
        if len(scx)==1:
            V.append(scx[0])
        else:
            E.append(scx)
    m,n=len(V),len(E)
    M=np.zeros((m,n),int)
    for i in range(m):
        for j in range(n):
            if V[i]==E[j][0]:
                M[i][j]=-1
            if V[i]==E[j][1]:
                M[i][j]=1
    return M


#Change k and especially l (number of cluster) for better results
def mapper(X,f,k,l,p):
    F=[f(x) for x in X]
    m,M=min(F),max(F)
    U=covering(m,M,k,p)
    print(U)
    pre_U=pre_image(F,U,X)
    C=[]
    
    for X_i in pre_U:
        C_i=K_means(X_i,l)  #Use clustering method of your choice
        C+=C_i

    G=graph(C)
    return G


#Persistant homology

def pivot(A,j): #Compute the pivot of the j-th column
    m=A.shape[0]
    for i in range(m):
        if A[i][j]==1:
            return i
    return None

def add_column(A,j,L):  #Add the column in L to the j-th column
    m=A.shape[0]
    for k in L:
        for i in range(m):
            A[i][j]+=A[i][k]
    Z2_coeff(A)

def Z2_coeff(A):
    size=A.shape
    m,n=size[0],size[1]
    for i in range(m):
        for j in range(n):
            if A[i][j]%2 ==0:
                A[i][j] = 0
            if A[i][j]%2 ==1 and A[i][j]<0:
                A[i][j] = -1
            if A[i][j]%2 ==1 and A[i][j]>0:
                A[i][j] = 1



X=[[1],[2],[3],[4],[5],[6],[2,3],[2,4],[1,2],[3,4],[1,3],[1,5],[2,5]]
A=boundary_matrix(X)
print(A)
print()
add_column(A,3,[0,1])
print(A)



"""
X=export("ant.ply")
print(X)
display(X)
print(mapper(X,d_from_x_axis,5,5,0.25))
#nearest_neighbor(X)
#clustering(X,0.2)
"""