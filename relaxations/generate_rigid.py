'''
Code written by Dr. Samuel Magorrian and Chung Xu
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as pth
from ase.calculators.lammpsrun import LAMMPS
from ase.optimize import BFGS;
from ase.io import write;
from ase import Atoms;
from os import environ;
from ase.io.lammpsdata import write_lammps_data, read_lammps_data;
import sys

m1 = sys.argv[1] 
m2 = sys.argv[2]
x1 = sys.argv[3]
stacking = sys.argv[4]

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))
def length(v):
  return math.sqrt(dotproduct(v, v))

# def angle(v1, v2):
#   return np.arccos(np.dot(v1, v2) / np.linalg.norm(np.dot(v1,v1) * np.dot(v2,v2)))
def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def inside_unit_cell(x,y,cell):
    n = len(cell)
    inside = False

    p1x,p1y = cell[0]
    for i in range(n+1):
        p2x,p2y = cell[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

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
#select from TMD values, interlayer distance currently arbitrary
a = 0.5*(a_optb88[f'{m1}{x1}2'] + a_optb88[f'{m2}{x1}2'])

#a = a_optb88[f'{m2}{x1}2']
dx = 0.5*(t[f'{m1}{x1}2']+t[f'{m2}{x1}2'])
interlayer_distance = 6.5
# Real primitive cell calculations
r3=np.sqrt(3)
a1 = [a, 0]
a2 = [0.5*a, r3*a/2]
# basis vector for non-rotated primitive cell
basis = [[0,0,0], [a/2,a/(2*r3),dx/2], [a/2,a/(2*r3),-dx/2]]

#print("a1", a1)
#print("a2", a2)
def get_angles_vectors(param):
    #in basis on original lattice
    ang_a1=[param, param+1]
    ang_a2=[param+1, param]

    ang_a1_xy=[ang_a1[0]*a1[0]+ang_a1[1]*a2[0],ang_a1[0]*a1[1]+ang_a1[1]*a2[1]]
    ang_a2_xy=[ang_a2[0]*a1[0]+ang_a2[1]*a2[0],ang_a2[0]*a1[1]+ang_a2[1]*a2[1]]

    rot_basis=[]

    twist_angle = angle(ang_a1_xy,ang_a2_xy)
    sina = math.sin(twist_angle)
    cosa = math.cos(twist_angle)

    for vector in basis:
        xnew = vector[0] * cosa - vector[1] * sina
        ynew = vector[0] * sina + vector[1] * cosa
        rot_basis.append([xnew, ynew, vector[2] + interlayer_distance])

    twist_angle_deg = twist_angle*(180/math.pi)
#     print(twist_angle_deg)
    sup_a1 = ang_a1
    sup_a2 = [ang_a1[0] + ang_a1[1], -ang_a1[0]]

    sup_a1_xy = [sup_a1[0] * a1[0] + sup_a1[1] * a2[0], sup_a1[0] * a1[1] + sup_a1[1] * a2[1]]
    sup_a2_xy = [sup_a2[0] * a1[0] + sup_a2[1] * a2[0], sup_a2[0] * a1[1] + sup_a2[1] * a2[1]]

    print("basis: ", basis)
    print("rot_basis: ", rot_basis)
    print(twist_angle_deg)
    return twist_angle_deg, rot_basis, [ang_a1, ang_a2], [ang_a1_xy, ang_a2_xy], [sup_a1_xy, sup_a2_xy], [cosa, sina]

def rotate_pcell(cossin):
    rot_a1 = [a1[0] * cossin[0], a1[0] * cossin[1]]
    rot_a2 = [a2[0] * cossin[0] -a2[1] * cossin[1], a2[0] * cossin[1] + a2[1] * cossin[0]]
    return rot_a1, rot_a2

def get_points(ang_a, ang_a_xy, sup_a_xy, cossin):
    
    points_x = []
    points_y = []

    rot_points_x = []
    rot_points_y = []
    
    # These 2 vectors are the rotated primitive cell vectors
    rot_a1, rot_a2 = rotate_pcell(cossin)
    
    n_points=(ang_a[0][0]**2 + ang_a[0][0] * ang_a[0][1] + ang_a[0][1]**2)
    print("total number of atoms: ", n_points * 6) #total number of atoms in twisted bilayer supercell

    super_cell=np.array([(0, 0),
        (sup_a_xy[0][0], sup_a_xy[0][1]),
    (sup_a_xy[0][0] + sup_a_xy[1][0], sup_a_xy[0][1] + sup_a_xy[1][1]),
    (sup_a_xy[1][0], sup_a_xy[1][1])]) + [(-0.1, -0.1)]

    nmax = 4 * max(ang_a[0])
    for n1 in range(-nmax, nmax + 1):
        for n2 in range(-nmax, nmax + 1):
            x=n1*a1[0]+n2*a2[0]
            y=n1*a1[1]+n2*a2[1]
            inn = inside_unit_cell(x, y, super_cell)
            if inn:
                points_x.append(x)
                points_y.append(y)
            x_r = n1*rot_a1[0] + n2*rot_a2[0]
            y_r = n1*rot_a1[1] + n2*rot_a2[1]
            inn_r = inside_unit_cell(x_r, y_r, super_cell)
            if inn_r:
                rot_points_x.append(x_r)
                rot_points_y.append(y_r)
                
    return [points_x, points_y], [rot_points_x, rot_points_y], n_points

def get_all_positions(points, rot_basis, rot_points, sup_a_xy, n_points,stacking):
    M = []
    X = []
    M2 = []
    X2 = []

    for point in range(0,n_points):
        M_11 = [basis[0][0] + points[0][point]-a/2, basis[0][1] + points[1][point] - a/(2*r3), basis[0][2]]
        X_11 = [basis[1][0] + points[0][point]-a/2, basis[1][1] + points[1][point] - a/(2*r3), basis[1][2]]
        X_12 = [basis[2][0] + points[0][point]-a/2, basis[2][1] + points[1][point] - a/(2*r3), basis[2][2]]

        M.append(M_11)
        X.append(X_11)
        X.append(X_12)

    # print(rot_basis)    
    for point in range(0,n_points):
        
        if stacking == 'P':
            
            M_21 = [rot_basis[0][0] + rot_points[0][point], basis[0][1] + rot_points[1][point], rot_basis[0][2]]
            X_21 = [rot_basis[1][0] + rot_points[0][point], basis[1][1] + rot_points[1][point], rot_basis[1][2]]
            X_22 = [rot_basis[2][0] + rot_points[0][point], basis[2][1] + rot_points[1][point], rot_basis[2][2]]
        elif stacking == 'AP':
            M_21 = [rot_basis[2][0] + rot_points[0][point], basis[2][1] + rot_points[1][point], rot_basis[0][2]]
            X_22 = [rot_basis[0][0] + rot_points[0][point], basis[0][1] + rot_points[1][point], rot_basis[2][2]]
            X_21 = [rot_basis[0][0] + rot_points[0][point], basis[0][1] + rot_points[1][point], rot_basis[1][2]]  
            
        M2.append(M_21)
        X2.append(X_21)
        X2.append(X_22)
        
    return M,X,M2,X2

z_thirty = [0, 0, 30]

def get_supercell_lattice_vectors(sup_a_xy):
    supercell_lattice_vectors = [sup_a_xy[0] + [0], sup_a_xy[1] + [0], z_thirty]
    print(supercell_lattice_vectors)
    return supercell_lattice_vectors

def create_bilayer(m1,m2,x,M,X, M2,X2, supercell_lattice_vectors):
    """
    Takes all positions and forms a TMDC encapsulated by an Atoms object
    """
    TM_layer1 = Atoms(m1*len(M), positions = M, cell = supercell_lattice_vectors, pbc = True)
    TM_layer2 = Atoms(m2*len(M2), positions = M2, cell = supercell_lattice_vectors, pbc = True)
    Chalcogen_layer1 = Atoms(x*len(X), positions = X, cell = supercell_lattice_vectors, pbc = True)
    Chalcogen_layer2 = Atoms(x*len(X2), positions = X2, cell = supercell_lattice_vectors, pbc = True)
    twisted_bilayer = TM_layer1  + TM_layer2 + Chalcogen_layer1 + Chalcogen_layer2
    nmetalatom = len(TM_layer1.positions)
    n_atom = len(twisted_bilayer.positions)
    return twisted_bilayer, nmetalatom, n_atom

def make_cell(m1,m2,x,ii,stacking):
    """
    Function which runs functions from all the above cells for one ii iteration
    """
    data = get_angles_vectors(ii)
    points = get_points(data[2], data[3], data[4], data[5])
    all_positions = get_all_positions(points[0], data[1], points[1], data[4], points[2],stacking)
    supercell_lattice_vectors = get_supercell_lattice_vectors(data[4])
    bilayer, nmetalatom, nallatom = create_bilayer(m1,m2,x,all_positions[0], all_positions[1], all_positions[2], all_positions[3], supercell_lattice_vectors)
    write_lammps_data(f"rigid_data/data_{m1}{x1}2_{m2}{x1}2_{stacking}_{ii}", bilayer)

for ii in range(40,41):
    make_cell(m1,m2,x1,ii,stacking)
