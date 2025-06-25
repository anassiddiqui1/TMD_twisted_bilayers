#!/bin/bash
m1=$1
x1=$2
m2=${m1}
x2=${x1}
align=$3
ii=$4
if [ "$m1" = "Mo" ]
then
   mass1=95.95
else
   mass1=183.84
fi

if [ "$x1" = "S" ]
then
   mass2=32.06
else
   mass2=78.971   
fi

if [ ${ii} -lt 3 ]
then
   partition="compute"
else
   partition="hmem"
fi

echo "ii=${ii} m1=${m1} x1=${x1} m2=${m2} x2=${x2} mass1=${mass1} mass2=${mass2} align=${align} partition=${partition}"
newDir="/home/theory/phrrpq/bilayer_relaxations/${m1}${x1}2_${m2}${x2}2/${align}/${ii}"
mkdir "${newDir}"
mkdir "${newDir}/restart"
cp "/home/theory/phrrpq/bilayer_relaxations/rigid_bilayers/data_${m1}${x1}2_${m2}${x2}2_${align}_${ii}" "${newDir}"
cat > "${newDir}/in.lammps" << EOF

clear
atom_style atomic
units metal
boundary p p p
atom_modify map yes

read_data ${newDir}/data_${m1}${x1}2_${m2}${x2}2_${align}_${ii}

### interactions
pair_style mace
pair_coeff * * /home/theory/phrrpq/bilayer_relaxations/MoWSSe_bi_radius_8_reduced_all_1_swa_800_run-123_swa.model-lammps.pt ${m1} ${x1}
mass 1 ${mass1}
mass 2 ${mass2}

### run
fix fix_nve all nve
restart 1 ${newDir}/restart/ii${ii}.*.restart
thermo_style custom step time cpu etotal fmax fnorm
thermo_modify flush yes format float %23.16g
thermo 1
timestep 1e-4
run 0
print "BEGIN LAMMPS STANDALONE MINIMIZATION; ii = ${ii}; OMP = 12; MPI = 4"
log /dev/stdout
minimize 0.0 1e-3 1000 100000
write_data ${newDir}/${m1}${x1}2_${m2}${x2}2_${align}_${ii}.data
EOF

cat > "${newDir}/run_lammps.sh" << EOF
#!/bin/bash
#SBATCH --partition=${partition}
#SBATCH -J ${m1}${x1}2_${m2}${x2}2_${align}_${ii}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH -o "${newDir}/${m1}${x1}2_${m2}${x2}2_${align}_${ii}.out"

module purge; module load GCC/11.3.0 OpenMPI/4.1.4 Python/3.10.4 PyTorch/1.12.1-CUDA-11.7.0 SciPy-bundle CMake

MY_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=\$MY_NUM_THREADS

echo "OMP_THREADS: \${OMP_NUM_THREADS}"
echo "MPI: 1"
echo "NTASKS-PER-NODE: \${SLURM_NTASKS_PER_NODE}"
echo "CPUS_PER_TASK: \${SLURM_CPUS_PER_TASK}"
mpirun -np 4 "/home/theory/phrrpq/lammps_mpi/build/lmp" -in "${newDir}/in.lammps"

EOF

sbatch "${newDir}/run_lammps.sh"
