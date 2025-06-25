import re
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.build.surface import mx2
from ase.visualize import view
import ase.units
import sys
from ase.io import read,write
from mace.calculators import MACECalculator
from ase.optimize import BFGS
from ase.constraints import FixedLine
calc = MACECalculator(model_paths="../MoWSSe_bi_radius_8_reduced_all_1_swa_800_run-123_swa.model",
                      device="cuda", default_dtype="float64")
from ase.lattice.hexagonal import Hexagonal
#calc = MACECalculator(model_paths="/home/theory/phrrpq/Research_Data/quaternary_test/bilayer_models/MoWSSe_bi_radius_8_reduced_all_1_swa_800_run-123_swa.model", 
#                      device="cuda", default_dtype="float64")
# #Opt88 relaxed lattice parameters
a = {}
a['WS2'] = 3.190902132
a['MoS2'] = 3.186305534
a['WSe2'] = 3.322874304
a['MoSe2'] = 3.322339468

t = {}
t['WS2'] = 3.1570497014
t['MoS2'] = 3.140170013
t['WSe2'] = 3.367178811
t['MoSe2'] = 3.3492559072

def passlog():
    pass

def make_TMDC(tm, ch, alat, clat, ch_height, index1, index2):
    mlat = Hexagonal(symbol=tm,latticeconstant={'a':alat,'c':clat},
                       size=(1,1,1))
    chlat1 = Hexagonal(symbol=ch,latticeconstant={'a':alat,'c':clat},
                       size=(1,1,1))
    chlat2 = Hexagonal(symbol=ch,latticeconstant={'a':alat,'c':clat},
                       size=(1,1,1))
    vec = chlat1.get_cell()
    chlat1.translate(vec[0]/3.0+2*vec[1]/3.0+[0,0,ch_height])
    chlat2.translate(vec[0]/3.0+2*vec[1]/3.0+[0,0,-ch_height])
    tmdc = mlat + chlat1 + chlat2
    return tmdc * (index1, index2, 1)

def make_bilayer(l1,l2, r0, interlayer_distance,stacking):
    monolayer_1 = l1.copy()
    monolayer_2 = l2.copy()
    if stacking == 'AP':
        monolayer_2.rotate(180,monolayer_2.positions[1]-monolayer_2.positions[2],
                     center=(monolayer_2.positions[1]+monolayer_2.positions[2])/2)
    monolayer_2.translate([r0[0], r0[1], interlayer_distance])
    return monolayer_1 + monolayer_2

N1 = 35
N2 = 25
m1 = sys.argv[1]
m2 = sys.argv[2]
x1 = sys.argv[3]
stacking = sys.argv[4]

print(f'Starting calculations for {stacking}-{m1}{x1}2/{m2}{x1}2')

a_tmd = 0.5*(a[f'{m1}{x1}2']+a[f'{m2}{x1}2'])
t_tmd = 0.5*(t[f'{m1}{x1}2']+t[f'{m2}{x1}2'])

l1_101 = make_TMDC(m1,x1,a_tmd, 20, t_tmd/2, 1, 2)
l1_101.wrap()
old_cell = l1_101.get_cell()
l1_rectangular_cell = l1_101.copy()
l1_rectangular_cell.set_cell([old_cell[0], old_cell[0] + old_cell[1], old_cell[2]])
l1_rectangular_cell.wrap()

l2_101 = make_TMDC(m2,x1,a_tmd, 20, t_tmd/2, 1, 2)
l2_101.wrap()
old_cell = l2_101.get_cell()
l2_rectangular_cell = l2_101.copy()
l2_rectangular_cell.set_cell([old_cell[0], old_cell[0] + old_cell[1], old_cell[2]])
l2_rectangular_cell.wrap()

offset_MX = a_tmd * np.array([0, 1 * 3**-0.5])
offset_XM = a_tmd * np.array([0, 2 * 3**-0.5])

if x1=='S':
    interd = 6.2
else:
    interd = 6.6

pure_MX = make_bilayer(l1_rectangular_cell,l2_rectangular_cell, offset_MX, interd,stacking)
pure_MX.calc = calc
dyn = BFGS(pure_MX)
dyn.run(fmax=0.001)
pure_MX_energy = pure_MX.get_potential_energy()

pure_XM = make_bilayer(l1_rectangular_cell,l2_rectangular_cell, offset_XM, interd,stacking)
pure_XM.calc = calc
dyn = BFGS(pure_XM)
dyn.run(fmax=0.001)
pure_XM_energy = pure_XM.get_potential_energy()
print(f"MM' and 2H energies : {pure_MX_energy} , {pure_XM_energy}")

folder_path = 'relaxed/'
pattern = re.compile(rf'{m1}{x1}2-{m2}{x1}2_{stacking}_N1_{N1}_N2_{N2}_constraint_(\d+)')
max_number = -1
max_file = None
for filename in os.listdir(folder_path):
    # Match the pattern to extract the number
    match = pattern.search(filename)
    if match:
        number = int(match.group(1))
        # Check if this number is the highest we've seen
        if number > max_number:
            max_number = number
            max_file = filename

combined_bilayer = read(f'relaxed/{max_file}')
natoms = len(combined_bilayer)
nunitcells = natoms//12


constraints_1 = []
constraints_2 = []
for index in range(natoms):
    if index<12*N1 or index>(natoms-12*N1):# or abs(index-natoms/2)<12*N1:
        if index%3==0:
            constraints_1.append(FixedLine(a=index,direction=[0, 0, 1]))
    else:
        if index%3==0:
            constraints_2.append(FixedLine(a=index,direction=[0, 0, 1]))

flag = True
k=max_number+1
prev_energy = 0
current_energy = 0
while flag:
    if k%2==0:
        combined_bilayer.set_constraint(constraints_1)
    else:
        combined_bilayer.set_constraint(constraints_2)

    combined_bilayer.calc = calc
    dyn = BFGS(combined_bilayer)
    #dyn.log = passlog
    dyn.run(fmax=0.001)

    write(f'relaxed/{m1}{x1}2-{m2}{x1}2_{stacking}_N1_{N1}_N2_{N2}_constraint_{k+1}.xyz',combined_bilayer)

    interlayer_d = []
    for i in range(0,natoms,12):
        l1_position = (combined_bilayer.positions[i]+combined_bilayer.positions[i+3])/2
        l2_position = (combined_bilayer.positions[i+6]+combined_bilayer.positions[i+9])/2
        interlayer_d.append(np.abs(l1_position-l2_position)[2])
    fig,axs = plt.subplots()
    axs.plot(interlayer_d)
    fig.savefig(f'figures/{m1}{x1}2-{m2}{x1}2_{stacking}_N1_{N1}_N2_{N2}_constraint_{k+1}.png')
    plt.close()

    cell_max_d_1,cell_max_d_2 = np.argmax(interlayer_d[:nunitcells//2]),np.argmax(interlayer_d[::-1][:nunitcells//2])
    ncells_MX = cell_max_d_1+cell_max_d_2+2
    ncells_XM = nunitcells-ncells_MX
    print(cell_max_d_1,cell_max_d_2,nunitcells,ncells_MX,ncells_XM)

    combined_bilayer_energy = combined_bilayer.get_potential_energy()
    print(f'Combined bilayer energy = {combined_bilayer_energy}')
    domain_wall_cost = (combined_bilayer_energy-ncells_MX*pure_MX_energy-ncells_XM*pure_XM_energy)/(2*np.linalg.norm(pure_MX.cell[1]))
    print(f'Energy cost per unit length : {domain_wall_cost}')

    if k==max_number+1:
        prev_energy = domain_wall_cost
        current_energy = domain_wall_cost
        energy_diff = abs(current_energy-prev_energy)
    else:
        prev_energy = current_energy
        current_energy = domain_wall_cost
        energy_diff = abs(current_energy-prev_energy)
        if k>7 and energy_diff<1e-4:
            flag=False

    print(f'Prev_energy - Current_energy = {energy_diff}')
    k+=1

