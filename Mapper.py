import matplotlib.pyplot as plt
import numpy as np
import numpy
import numpy.linalg
import math
import pandas as pd
from sklearn.cluster import KMeans
import networkx as nx
from itertools import combinations



# Functions f (can be a projection)

def d(x, y):
    """Return the Euclidean distance between the points x and y

    Parameters
    ----------
    x : list
        The first point
    y : list
        The second point

    Returns
    -------
    float
        The Euclidean distance between the points x and y"""
    y = [0, 0, 0]
    n = len(x)
    S = 0
    for i in range(n):
        S += (x[i] - y[i])**2
    return math.sqrt(S)

def d_from_x_axis(p):
    """Return the distance of the point p from the x-axis

    Parameters
    ----------
    p : list
        The point to compute the distance from the x-axis

    Returns
    -------
    float
        The distance of the point p from the x-axis"""
    return d(p, [p[0], 0, 0])

def d_from_y_axis(p):
    """Return the distance of the point p from the y-axis
    
    Parameters
    ----------
    p : list
        The point to compute the distance from the y-axis
        
    Returns
    -------
    float
        The distance of the point p from the y-axis"""
    return d(p, [0, p[1], 0])

def d_from_z_axis(p):
    """Return the distance of the point p from the z-axis
    
    Parameters
    ----------
    p : list
        The point to compute the distance from the z-axis
        
    Returns
    -------
    float
        The distance of the point p from the z-axis"""
    return d(p, [0, 0, p[2]])

def proj_x(p):
    """Return the projection of the point p on the x-axis
    
    Parameters
    ----------
    p : list
        The point to project
        
    Returns
    -------
    float
        The projection of the point p on the x-axis"""
    return p[0]

def proj_y(p):
    """Return the projection of the point p on the y-axis
    
    Parameters
    ----------
    p : list
        The point to project
        
    Returns
    -------
    float
        The projection of the point p on the y-axis"""
    return p[1]

def proj_z(p):
    """Return the projection of the point p on the z-axis
    
    Parameters
    ----------
    p : list
        The point to project
    
    Returns
    -------
    float
        The projection of the point p on the z-axis"""
    return p[2]

#Covering U
def covering(m, M, k, p):
    """Return the covering of the interval [m,M] with k spans of length l=(M-m)/k*(1+p) with p percentage of overlapping
    
    Parameters
    ----------
    m : float
        The minimum of the interval
    M : float
        The maximum of the interval
    k : int
        The number of spans
    p : float
        The percentage of overlapping
        
    Returns
    -------
    cover : list of tuples
        The covering of the interval [m,M] with k spans of length l=(M-m)/k*(1+p) with p percentage of overlapping"""
    l = (M - m) / k * (1 + p)
    overlap = l*p
    cover = []
    a = m
    b = m + l
    for i in range(k):
        cover.append((a, b))
        a = b - overlap
        b = a + l
    return cover

def in_interval(x, U):
    """Return True iff x is in the interval U
    
    Parameters
    ----------
    x : float
        The point to test
    U : tuple
        The interval
    
    Returns
    -------
    bool
        True iff x is in the interval U"""
    return (x >= U[0]) and (x <= U[1])

def pre_image(Y, U, X):
    """Return the pre-image of the intervals U in the list Y (f^(-1)(U_i))
    
    Parameters
    ----------
    Y : list
        The list of images
    U : list of tuples
        The intervals
    X : list
        The list of points
        
    Returns
    -------
    L : list of lists
        The pre-image of the intervals U in the list Y"""
    L = []
    n = len(Y)
    for u in U:
        L_i = []
        for i in range(n):
            if in_interval(Y[i], u):
                L_i.append(X[i])
        L.append(L_i)
    return L

def intersection(A, B):
    """Return the intersection of the lists A and B

    Parameters
    ----------
    A : list
        The first list
    B : list
        The second list

    Returns
    -------
    L : list
        The intersection of A and B"""
    L = []
    for a in A:
        if a in B:
            L.append(a)
    return L

def graph(clusters):
    """Return the graph of the clusters

    Parameters
    ----------
    clusters : list of lists
        The clusters to compute the graph of

    Returns
    -------
    G : list of lists
        The graph of the clusters"""
    k = len(clusters)
    G = [[i] for i in range(k)]
    for i in range(k-1):
        for j in range(i+1, k):
            if intersection(clusters[i], clusters[j]) != []:
                G.append([i, j])
    return G


def export(file):
    """Return the list of points in the ply file

    Parameters
    ----------
    file : str
        The path to the file to read

    Returns
    -------
    points : list of lists
        The points in the file"""
    with open(file, "r") as f:
        line = None
        while line != 'end_header\n':
            line = f.readline()
        
        points = []
        while True:
            line = f.readline()[:-2]
            nbrs = line.split(' ')
            if len(nbrs) == 3:
                points.append(tuple(float(num) for num in nbrs))
            else:
                break
    return points




def convert(X):
    """Convert the list of points X into a DataFrame

    Parameters
    ----------
    X : list of lists
        The points to convert
        
    Notes
    -----
    X is changed during the process"""
    X = pd.DataFrame(X, columns = ['abscisse','ordonnée'])

def K_means(X, k):
    """Return the clusters of the data points in X using the K-means algorithm

    Parameters
    ----------
    X : list of lists
        The data points to cluster
    k : int
        The number of clusters

    Returns
    -------
    C : list of lists
        The clusters"""
    clus = KMeans(n_clusters = k, random_state = 0, n_init = "auto").fit_predict(X)
    n = len(X)
    C = [[] for i in range(k)]
    for i in range(n):
        j = clus[i]
        C[j].append(X[i])
    return C

def barycenter(points):
    """Return the barycenter of the points
    
    Parameters
    ----------
    points : list of lists
        The points to compute the barycenter of
        
    Returns
    -------
    S : list
        The barycenter"""
    dim = len(points[0])
    n = len(points)
    S =[0.] * dim
    for x in points:
        for i in range(dim):
            S[i] += x[i]
    for j in range(dim):
        S[j] = S[j] / n
    return S

def centers(clusters):
    """Return the list of centers of the clusters
    
    Parameters
    ----------
    clusters : list of lists
        The clusters
    
    Returns
    -------
    C : list of lists
        The list of centers"""
    C = []
    for clus in clusters:
        C.append(barycenter(clus))
    return C

def display(X):
    """
    Display the data points in X
    
    Parameters
    ----------
    X : list of lists
        The data points to display"""
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.scatter([x[0] for x in X], [x[1] for x in X], [x[2] for x in X], label = 'Courbe', marker = 'd')
    plt.show()

def boundary_matrix(X):
    """
    Return the boundary matrix of the 2-dim simplicial complex X
    
    Parameters
    ----------
    X : list of lists
        The 2-dim simplicial complex
        
    Returns
    -------
    M : numpy array
        The boundary matrix of X"""
    V = []
    E = []
    for scx in X:
        if len(scx) == 1:
            V.append(scx[0])
        else:
            E.append(scx)
    m, n = len(V), len(E)
    M = np.zeros((m, n), int)
    for i in range(m):
        for j in range(n):
            if V[i] == E[j][0]:
                M[i][j] = -1
            if V[i] == E[j][1]:
                M[i][j] = 1
    return M


def mapper(X, f, k, l, p):
    """Return the graph of the mapper complex of the data X
    
    Parameters
    ----------
    X : list of points
        Data
    f : function
        Function to apply to the data
    k : int
        Number of spans
    l : int
        Number of clusters
    p : float
        Percentage of overlapping
        
    Returns
    -------
    list of list of int
        Graph of the mapper complex
        
    Notes
    -----
    Change k and especially l (number of cluster) for better results"""
    F = [f(x) for x in X]
    m, M = min(F), max(F)
    U = covering(m, M, k, p)
    pre_U = pre_image(F, U, X)
    C = []
    
    for X_i in pre_U:
        C_i = K_means(X_i, l)  #Use clustering method of your choice
        C += C_i

    G=graph(C)
    return G

##### PERSISTANT HOMOLOGY #####

def pivot(A, j):
    """Return the index of the pivot of the j-th column

    Parameters
    ----------
    A : numpy array
        Matrix of the boundary matrix
    j : int
        Index of the column
        
    Returns
    -------
    int
        Index of the pivot"""
    m = A.shape[0]
    for i in range(m):
        if A[i][j] == 1:
            return i
    return None

def add_column(A, j, L):
    """Add the column in L to the j-th column
    
    Parameters
    ----------
    A : numpy array
        Matrix of the boundary matrix
    j : int
        Index of the column
    L : list
        List of the index of the columns to add
        
    Notes
    -----
    A is changed during the process"""
    m = A.shape[0]
    for k in L:
        for i in range(m):
            A[i][j] += A[i][k]
    Z2_coeff(A)

def is_canceled(A, j, L):
    """Return True if adding the column of list L to the j-th column make it a zero column
    
    Parameters
    ----------
    A : numpy array
        Matrix of the boundary matrix
    j : int
        Index of the column
    L : list
        List of the index of the columns to add
        
    Returns
    -------
    bool
        True if adding the column of list L to the j-th column make it a zero column"""
    B=A.copy()
    add_column(B,j,L)
    m=B.shape[0]
    for i in range(m):
        if B[i][j] != 0:
            return False
    return True

def is_zero_column(A, j):
    """Return True if the j-th column of A is null
    
    Parameters
    ----------
    A : numpy array
        Matrix of the boundary matrix
    j : int
        Index of the column

    Returns
    -------
    bool
        True if the j-th column of A is null
    """
    m = A.shape[0]
    for i in range(m):
        if A[i][j] != 0:
            return False
    return True

def combination(L):
    """Return all the possible combination with elements of L
    
    Parameters
    ----------
    L : list
        List of elements
        
    Returns
    -------
    C : list
        List of all the possible combination with elements of L"""
    C = []
    for k in range(1, len(L) + 1):
        C += list(combinations(L,k))
    return C

def Z2_coeff(A):
    """Change the coefficient of the matrix A to be in Z2
    
    Parameters
    ----------
    A : numpy array
        Matrix of the boundary matrix
        
    Notes
    -----
    The matrix A is changed"""
    size = A.shape
    m, n = size[0], size[1]
    for i in range(m):
        for j in range(n):
            if A[i][j] % 2 == 0:
                A[i][j] = 0
            if A[i][j] % 2 == 1 and A[i][j] < 0:
                A[i][j] = -1
            if A[i][j] % 2 == 1 and A[i][j] > 0:
                A[i][j] = 1

def reduction(A):
    """Reduce the matrix A to its reduced row echelon form

    Parameters
    ----------
    A : numpy array
        Matrix of the boundary matrix
        
    Notes
    -----
    The matrix A is changed"""
    n = A.shape[1]
    non_zero_column = [0]
    for j in range(1, n):
        comb = combination(non_zero_column)
        canceled = False
        for L in comb:
            if is_canceled(A, j, L):
                add_column(A, j, L)
                canceled = True
                break
        if not canceled:
            non_zero_column.append(j)
        
def paired_vertices(A):
    """Return the list of birth-terminal pairs and unpaired vertices
    
    Parameters
    ----------
    A : numpy array
        Matrix of the boundary matrix
        
    Returns
    -------
    L : list of list
        List of birth-terminal pairs and unpaired vertices"""
    m, n = A.shape[0], A.shape[1]
    L=[]
    for j in range(n):
        paired = False
        for i in range(m):
            if A[i][j] == 1:    #Pivot
                L.append([1, j + 2])
                paired = True
                break
        if not(paired):
            L.append([1, "inf"])
    return L

def show_persistant_diagramm(paired):
    """Display the persistant diagramm of the list of birth-death pairs
    
    Parameters
    ----------
    paired : list of list
        List of birth-death pairs"""
    fig = plt.figure()
    n = len(paired)
    X = np.linspace(0, n + 1.5, 1000)
    axes = plt.gca()
    axes.xaxis.set_ticks(range(n + 2))
    axes.yaxis.set_ticks(range(n + 2))
    grad = [i for i in range(n+1)]
    axes.yaxis.set_ticklabels(grad + ["$\infty$"])
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.plot(X, X)
    for p in paired:
        if p[1] == "inf":
            plt.scatter(p[0], n + 1, c = 'r', label = 'H1')
        else:
            plt.scatter(p[0], p[1], c = 'r', label = 'H1')
    for i in range(2,n + 1):
        plt.scatter(i, n + 1, c = 'b', label = 'H0')
    plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Persistent Diagram")
    plt.show()

def show_graph(data):
    """Plot the graph of the input data.
    
    Parameters
    ----------
    data : list of list
        The input data. Each sublist of len 1 is a node, each sublist of len 2 is an edge."""
    G = nx.Graph()
    nodes = set().union(*data)
    edges = [tuple(edge) for edge in data if len(edge) == 2]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges) 
    nx.draw(G, with_labels=True)
    plt.show()




file = "ant.ply"
X = export(file)
display(X)
mapper_data = mapper(X,d_from_x_axis,6,3,0.25)
A = boundary_matrix(mapper_data)
reduction(A)
show_graph(mapper_data)
L = paired_vertices(A)
show_persistant_diagramm(L)
