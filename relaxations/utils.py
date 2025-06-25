import math
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read,write;
from ase import Atoms;
from ase.build.surface import mx2
from os import environ;
# from ase.io.lammpsdata import write_lammps_data, read_lammps_data;
import sys
from glob import glob
from ase.visualize import view
from curses import meta
import math;
import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib as mpl;
from ase import Atoms;
from ase.io import write, read;
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from scipy.spatial.distance import cdist

# nmetalatoms is a number which allows us to extract the positions of all metal atoms (see previous cells)
def extract_positions(relaxed_structure):
    """
    relaxed_structure: Atoms object
    nmetalatom: int. Number of metal atoms
    ===
    
    Assumes the first nmetalatom entries of the xyz file are metal and extracts their positions
    
    ===
    Returns: the positions of the metal atoms in each layer
    """
    nmetalatom = len(relaxed_structure)//6
    metal1_relaxed_pos = np.array(relaxed_structure.positions[0:nmetalatom], dtype="float64")
    metal2_relaxed_pos = np.array(relaxed_structure.positions[nmetalatom:nmetalatom*2], dtype="float64")
    return metal1_relaxed_pos, metal2_relaxed_pos

### extend coordinates around simulation cell
def ExtendCoordinates(x, y, z, cell):
    '''
    Extend coordinates around simulation cell to avoid artifacts during interpolation

    Parameters
    ----------
    x, y, z : list
        Atomic coordinates within a single layer, split into x-, y- and z-component.
    cell : 2D list
        Simulation cell vectors: [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]].

    Returns
    -------
    x_sc, y_sc, z_sc : list
        Extended coordinates, split into x-, y- and z-component.
    '''
    ## preparation
    # lists for new values
    x_sc, y_sc, z_sc = [], [], [] # create new lists
    x_sc_frac, y_sc_frac = [], [] # create new lists
    # transformation matrix to obtain fractional coordinates
    A = np.array([[cell[0][0], cell[1][0]], [cell[0][1], cell[1][1]]])
    # calculate what fractional value DIST_EXTEND corresponds to
    FRAC_CUTOFF = DIST_EXTEND / np.sqrt(cell[0][0]**2+cell[0][1]**2+cell[0][2]**2)
    ## make 3x3 supercell from initial coordinates
    for i in range(len(x)): # over all atoms
        # calculate fractional coordinates
        vec_x = np.array([x[i], y[i]]) # coordinate vector
        [x_frac, y_frac] = np.linalg.solve(A, vec_x) # get fractional coordinates
        # apply supercell
        for m in [-1,0,1]: # supercell in x-direction
            for n in [-1,0,1]: # supercell in y-direction
                x_test = x_frac + m
                y_test = y_frac + n
                # filter out what needed
                if x_test>=-FRAC_CUTOFF and x_test<=(1+FRAC_CUTOFF):
                    if y_test>=-FRAC_CUTOFF and y_test<=(1+FRAC_CUTOFF):
                        x_sc_frac.append(x_test)
                        y_sc_frac.append(y_test)
                        z_sc.append(z[i])
    ## go back from fractional to cartesian coordinates
    for i in range(len(x_sc_frac)): # over all coordinates
        x_sc.append(x_sc_frac[i]*cell[0][0]+y_sc_frac[i]*cell[1][0])
        y_sc.append(x_sc_frac[i]*cell[0][1]+y_sc_frac[i]*cell[1][1])  
    ## return statement
    return x_sc, y_sc, z_sc

import numpy as np

def MakeGrid(cell, theta, A_ML, high_res=False):
    if theta > 30:
        theta = 60 - theta  # enforce symmetry

    # Moiré cell constant (approximation)
    a_moire = A_ML / np.sqrt(2 * (1 - np.cos(np.deg2rad(theta))))

    # Determine A_REF based on resolution mode
    if high_res:
        if a_moire <= 23:
            A_REF = 20  # tiny cell size
        elif a_moire <= 65:
            A_REF = 50  # small cell size
        elif a_moire <= 250:
            A_REF = 200  # intermediate size
        else:
            A_REF = 1000  # large
    else:
        if a_moire <= 20:
            A_REF = 20
        elif a_moire <= 50:
            A_REF = 50
        elif a_moire <= 250:
            A_REF = 200
        else:
            A_REF = 1000

    # Lattice constant from unit cell vector
    a = np.sqrt(cell[0][0]**2 + cell[0][1]**2 + cell[0][2]**2)

    # Determine grid resolution using linear fit
    GRID_STEPS = int(77.832 * (a / A_REF) + 9.8418) + 1

    # Generate grid points
    x_grid, y_grid = [], []
    for i in range(GRID_STEPS):
        for k in range(GRID_STEPS):
            x = i * cell[0][0] / GRID_STEPS + k * cell[1][0] / GRID_STEPS
            y = i * cell[0][1] / GRID_STEPS + k * cell[1][1] / GRID_STEPS
            x_grid.append(x)
            y_grid.append(y)

    return x_grid, y_grid, A_REF


### perform interpolation of z-coordinates
def DoInterpolation(x_grid, y_grid, x_ext, y_ext, z_ext):
    '''
    Perform interpolation of extended coordinates on a given grid

    Parameters
    ----------
    x_grid, y_grid : list
        Interpolation grid with separate x- and y-coordinates.
    x_ext, y_ext, z_ext : list
        Coordinates, extended around simulation cell, separated x-, y-, and z-components.

    Returns
    -------
    z_int : list
        Interpolated z-coordinates, corresponding to grid points.
    '''
    ## interpolate using griddata method from SciPy
    z_int = griddata(points=(x_ext, y_ext), values=z_ext, xi=(x_grid, y_grid), method=GRIDDATA_ORDER)
    ## return statement
    return z_int


### calculate local interlayer distance
def GetInterlayerDistance(z_UL, z_LL):
    '''
    Calculate interlayer distance from interpolated coordinates

    Parameters
    ----------
    z_UL, z_LL : list
        Interpolated z-coordinates, corresponding to grid points.

    Returns
    -------
    d_local : list
        Local interlayer distance at grid points.
    '''
    ## calculate property
    d_local = []
    for i in range(len(z_UL)): # for all grid points
        d_local.append(z_UL[i] - z_LL[i]) # d = z(UL) - z(LL)
    ## return statement
    return d_local



###Stuff to contact nearest neighbors
from scipy.spatial import cKDTree
from itertools import combinations

def calculate_distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def find_nearest_neighbors(x, y, k=2):
    points_array = list(zip(x, y))
    kdtree = cKDTree(points_array)
    _, indices = kdtree.query(points_array, k=k)
    return indices

def is_equilateral_triangle(triangle_points):
    # Define your equilateral triangle criteria here
    # For example, check if all side lengths are equal
    side_lengths = [calculate_distance(triangle_points[i], triangle_points[j]) for i, j in [(0, 1), (1, 2), (2, 0)]]
    #print(side_lengths,all(round(length,2) == round(side_lengths[0],2) for length in side_lengths))
    
    return all(round(length,1) == round(side_lengths[0],1) for length in side_lengths)

def find_equilateral_triangles(x, y, k=3):
    nearest_neighbors = find_nearest_neighbors(x, y, k=k)
    equilateral_triangles = []

    # Generate all combinations of 3 points among nearest neighbors
    for i, neighbors in enumerate(nearest_neighbors):
        #print(neighbors)
        for neighbor_combination in combinations(neighbors[1:],2):  # Fix here
            
            triangle_points = [(x[i], y[i])] + [(x[j], y[j]) for j in neighbor_combination]
            if is_equilateral_triangle(triangle_points):
                #print(neighbor_combination)
                equilateral_triangles.append(triangle_points)

    return np.asarray(equilateral_triangles)

def get_centroids(triangles):
    
    centroids = np.sum(triangles,axis=1)/3
    return centroids[:,0],centroids[:,1]

def rotate_vector(vector, angle):
    # Convert the angle to radians
    angle_rad = np.radians(angle)

    # Create a rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])

    # Perform the 2D rotation using matrix multiplication
    rotated_vector = np.dot(rotation_matrix, vector)

    return rotated_vector

def rotate_vector_3d(vector, angle):
    # Extract x and y components from the 3D vector
    x, y, _ = vector

    # Create a 2D rotation matrix
    rotation_matrix_2d = np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]])

    # Perform the 2D rotation using matrix multiplication
    rotated_vector_2d = np.dot(rotation_matrix_2d, np.array([x, y]))

    # Append a zero to represent the z component
    rotated_vector_3d = np.append(rotated_vector_2d, 0)

    return rotated_vector_3d

def scale_layer(atoms,indices,scale,mean_increase=0):
    
    atom_positions = atoms.positions[indices]
    z_positions = atom_positions[:,2]
    layer_mean = np.mean(z_positions)
    diff_positions = z_positions-layer_mean
    atoms.positions[indices,2] = layer_mean + mean_increase+ scale*diff_positions

def scale_atoms(atoms,scale,mean_increase=0):
    nmetalatom = len(atoms)//6
    m_l1_indices = np.arange(0,nmetalatom)
    s1_l1_indices = np.arange(2*nmetalatom,4*nmetalatom,2)
    s2_l1_indices = np.arange(2*nmetalatom+1,4*nmetalatom,2)
    m_l2_indices = np.arange(nmetalatom,nmetalatom*2)
    s1_l2_indices = np.arange(4*nmetalatom,6*nmetalatom,2)
    s2_l2_indices = np.arange(4*nmetalatom+1,6*nmetalatom,2)
    
    for l in ['l1','l2']:
        for atype in ['m','s1','s2']:
            if l=='l2' and mean_increase!=0:
                elevate = mean_increase
            else:
                elevate=0
            scale_layer(atoms,eval(f'{atype}_{l}_indices'),scale,elevate)

def rotate_view(vw, x=0, y=0, z=0, degrees=True):
    radians = 1
    if degrees: radians = np.pi / 180
    vw.view.control.spin([1, 0, 0], x*radians)
    vw.view.control.spin([0, 1, 0], y*radians)
    vw.view.control.spin([0, 0, 1], z*radians)
    
    
def get_max_ds(x_sc,y_sc,d_local_sc,x_min,x_max,y_min,y_max,bool_x,spacing=100):
    
    xdata = []
    ydata = []

    xspace = np.linspace(x_min,x_max,spacing)
    yspace = np.linspace(y_min,y_max,spacing)
    xs,ys = np.meshgrid(xspace,yspace)
    ds = griddata(points=(x_sc, y_sc), values=d_local_sc, xi=(xs, ys), method=GRIDDATA_ORDER)
    
    if bool_x:
        for i,x in enumerate(xspace):
            max_arg = np.argmax(ds[:,i])
            xdata.append(xs[max_arg][i])
            ydata.append(ys[max_arg][i])
    else:
        for i,x in enumerate(xspace):
            max_arg = np.argmax(ds[i,:])
            xdata.append(xs[i][max_arg])
            ydata.append(ys[i][max_arg])
        
    return xdata,ydata

def get_closest_point(x_ref,y_ref,xs,ys):
    distances = np.sqrt((xs - x_ref)**2 + (ys - y_ref)**2)
    closest_index = np.argmin(distances)
    return xs[closest_index],ys[closest_index]

def calculate_slope(x1, y1, x2, y2):
    if x2!=x1:
        return (y2 - y1) / (x2 - x1)
    else:
        return float('inf')

# Find the negative reciprocal of the slope
def get_perpendicular_slope(slope):
    if slope!=float('inf'):
        return -1 / slope
    else:
        return 0

# Calculate the y-intercept of the line passing through a given point with a given slope
def calculate_intercept(x, y, slope):
    return y - slope * x

def find_line_parameters(x1, y1, x2, y2):
    # Calculate the slope (m)
    m = (y2 - y1) / (x2 - x1)
    
    # Calculate the y-intercept (b)
    b = y1 - m * x1
    
    return m, b

def get_tangent(x1,x2,y1,y2,xrange):
    slope = calculate_slope(x1, y1, x2, y2)
    perpendicular_slope = get_perpendicular_slope(slope)
    intercept = calculate_intercept(x2,y2,perpendicular_slope)
    xs = np.linspace(x2-xrange,x2+xrange,10)
    ys = perpendicular_slope*xs+intercept
    return xs,ys,perpendicular_slope,intercept

# slope_23 = calculate_slope(x_cent, y_cent, x_closest_23, y_closest_23)
# perpendicular_slope_23 = get_perpendicular_slope(slope_23)
# y_intercept_23 = calculate_y_intercept(x_closest_23, y_closest_23, perpendicular_slope_23)

def mirror_image(xs, ys, x1=None, y1=None):
    # Convert the input arrays to NumPy arrays
    mirrored_xs = xs.copy()
    mirrored_ys = ys.copy()
    
    if x1 is not None:
        # Calculate the distance between each point and the line x = x1
        distances = mirrored_xs - x1
        # Mirror image calculation
        mirrored_xs = 2 * x1 - mirrored_xs
    
    if y1 is not None:
        # Calculate the distance between each point and the line y = y1
        distances = mirrored_ys - y1
        # Mirror image calculation
        mirrored_ys = 2 * y1 - mirrored_ys
        
    return mirrored_xs, mirrored_ys

def rotate_points(xs, ys, angle, x1, y1):
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Translate the points so that the pivot point is at the origin
    xs_translated = xs - x1
    ys_translated = ys - y1
    
    # Compute the rotated coordinates using the rotation matrix
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    xs_rotated = xs_translated * cos_angle - ys_translated * sin_angle
    ys_rotated = xs_translated * sin_angle + ys_translated * cos_angle
    
    # Translate the rotated points back to their original position
    xs_rotated += x1
    ys_rotated += y1
    
    return xs_rotated, ys_rotated

def get_grid_data(atoms,a_lat,twist_angle,hres=False):
    
    cell = atoms.cell
    coord_LL,coord_UL = extract_positions(atoms)
    x_UL, y_UL, z_UL = coord_UL[:,0], coord_UL[:,1], coord_UL[:,2]
    x_LL, y_LL, z_LL = coord_LL[:,0], coord_LL[:,1], coord_LL[:,2]

    # extend coordinates to avoid interpolation artifacts
    x_UL_ext, y_UL_ext, z_UL_ext = ExtendCoordinates(x_UL, y_UL, z_UL, cell)
    x_LL_ext, y_LL_ext, z_LL_ext = ExtendCoordinates(x_LL, y_LL, z_LL, cell)
    # calculate interlayer distance
    x_grid, y_grid, A_REF = MakeGrid(cell, twist_angle,a_lat,hres=hres) # create interpolation grid
    z_UL_int = DoInterpolation(x_grid, y_grid, x_UL_ext, y_UL_ext, z_UL_ext) # interpolate UL
    z_LL_int = DoInterpolation(x_grid, y_grid, x_LL_ext, y_LL_ext, z_LL_ext) # interpolate LL
    d_local = GetInterlayerDistance(z_UL_int, z_LL_int) # calculate interlayer distance
    
    X_NEG, X_POS = A_REF, 2*A_REF
    Y_NEG, Y_POS = int(0.7*A_REF), int(1.3*A_REF)
    # get cell constants
    a = np.sqrt(cell[0][0]**2+cell[0][1]**2+cell[0][2]**2) # cell constant a
    b = np.sqrt(cell[1][0]**2+cell[1][1]**2+cell[1][2]**2) # cell constant b
    # find how much to extend in terms of supercells
    x_left = math.ceil((X_NEG + 0.5*a)/a)+2
    x_right = math.ceil(X_POS/a)+1
    y_bottom = math.ceil(Y_NEG/(SIN60*b))+1
    y_top = math.ceil(Y_POS/(SIN60*b))+1

    ## prepare interpolated data: extend within supercell of plotting area
    x_sc, y_sc, d_local_sc = [], [], [] # prepare lists
    for m in range(-x_left-1, x_right+1): # supercell in x-direction
        for n in range(-y_bottom-1, y_top+1): # supercell in y-direction
            for i in range(len(d_local)): # over all data points
                x_sc.append(x_grid[i]+m*cell[0][0]+n*cell[1][0]) # x coordinates
                y_sc.append(y_grid[i]+m*cell[0][1]+n*cell[1][1]) # y coordinates
                d_local_sc.append(d_local[i]) # local interlayer distance
    
    return x_sc,y_sc,d_local_sc,A_REF,X_NEG,X_POS,Y_NEG,Y_POS

FIXED_COLORBAR = False # toggle if colorbar should have fixed range (set D_MIN and D_MAX accordingly) or a variable range (based on structure)
## interpolation settings
DIST_EXTEND = 35 # Ang, value to extend around simulation cell for interpolation
GRIDDATA_ORDER = 'cubic' # order of griddata interpolation. options: 'nearest', 'linear', 'cubic'
## plotting
SIZE_PT = 0.4 # size of points for scatter plot, this value gave generally good results

DPI = 300 # dpi for figure export
## numerics
SIN60 = np.sin(60/360*2*np.pi) # often needed for b axis of unit cell
COS60 = np.cos(60/360*2*np.pi) # for nicer looking code

#A_ML = 3.186305534

#Opt88 relaxed lattice parameters
a_optb88 = {}
a_optb88['WS2'] = 3.190902132
a_optb88['MoS2'] = 3.186305534
a_optb88['WSe2'] = 3.322874304
a_optb88['MoSe2'] = 3.322339468

t = {}
t['WS2'] = 3.1570497014
t['MoS2'] = 3.140170013
t['WSe2'] = 3.367178811
t['MoSe2'] = 3.3492559072

twist_angles = {1:21.786789298261844,2:13.173551107258918,3:9.430007907896421,
                4:7.340993016630217,5:6.008983197766087,8:3.890238169007777,
                9:3.4810060894667294,10:3.1496574263895005,
                20:1.6135389011623444,30:1.0845490491576433,35:0.9318029472653259,
                40:0.8167697168934015,15:2.1339296665974214,16:2.004627830691014,
                17:1.8900990734698215,18:1.787948610382325,11:2.875894633632844,
               12:2.645908381193102,13:2.4499772766168695,14:2.2810596097298803
                ,15:2.1339296665974214,19:1.6962726935197672,22:1.4701297257774135,
               25:1.2971890475932204,27:1.202855227481003}