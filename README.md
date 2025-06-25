# Understanding Domain Reconstruction of Twisted Transition Metal Dichalcogenides Bilayers through Machine Learned Interatomic Potentials

## 📂 Folder & File Descriptions

### `training/`

This folder contains the training, validation and test sets, the slurm script to train the MACE model, and the model itself.

- **`MoWSSe_bi_train.xyz`**: Training split of the dataset
- **`MoWSSe_bi_vald.xyz`**: Validation split of the dataset
- **`MoWSSe_bi_test.xyz`**: Test split of the dataset
- **`run_mace`**: Slurm script used to train on Sulis HPC
- **`training.log`**: The log generated during the training
- **`MoWSSe_bi_radius_8_reduced_all_1_swa_800_run-123_swa.model`**: The trained model
- **`MoWSSe_bi_radius_8_reduced_all_1_swa_800_run-123_swa.model-lammps.pt`**: The trained model for LAMMPS usage

### `testing/`
This directory contains scripts and notebooks for testing various aspects of the models and simulations, including adhesive tests, `espresso` (likely Quantum ESPRESSO) related tests for relaxation and strain, and parameter tests.
- **`primitive`**: (directory) Contains DFT relaxation for the primitive cell of monolayers.
- **`adhesive_test.ipynb`**: Jupyter Notebook for comparing DFT vs MLIP adhesive energy densities.
  This has been done on training (energy_vs_d_dft_mace_train.png) as well as test (energy_vs_d_dft_mace_test.png) configurations and produces Figures S2, S3 and Tables I, S2, S4.
- **`aligned_test`**: (directory) Contains DFT scf calculations for aligned bilayers in the training dataset.
- **`aligned_train`**: (directory) Contains DFT scf calculations for aligned bilayers in the test dataset where they have random stacking offsets.
- **`espresso_relax_train`**: (directory) Contains DFT constrained relaxations for stackings used during training.
- **`espresso_relax_test`**: (directory) Contains DFT constrained relaxations for stackings used during testing.
- **`strain_test.ipynb`**: Jupyter Notebook for comparing DFT vs MLIP strain energetics of pure monolayers. Generates 'strain_test.png', which is used as Fig. S3.
- **`espresso_strain`**: (directory) DFT scf calculations on strained configurations of pure monolayers used in strain_test.ipynb.
- **`lat_param_test.ipynb`**: Jupyter Notebook for comparing DFT vs MLIP heterobilayer energetics as a function of lattice parameter used. Generates 'lat_param_test.png', which is used as Fig. S4.
- **`alat_espresso`**: (directory) DFT scf calculations on heterobilayers with different lattice parameters used in lat_param_test.ipynb


### `energy_model/`
This directory likely contains scripts and data related to the energy model, possibly for calculating or analyzing the energy of the system.

- **`dwallcost`**: (directory) Contains files to run domain wall cost calculations used in energy model and shown in Table 2 and 3.
'domain_wall_cost.py' that runs the constrained optimization, 'read_dwall_cost.ipynb' to interpret the output file, and run_mace slurm script to run GPU calculation on sulis HPC.
- **`output_files`**: (directory) Contains LAMMPS output files for all relaxations to read total energy.
- **`total_energy_model.ipynb`**: Jupyter Notebook for the total energy model, which is used to calculate node energies
as a function of twist angle and generates dictionary 'node_energies.pkl' and image 'node_energies.png', which is used as Figure S12.
It is also used to generate image 'binding_e_vs_twist.png', which is used as Figure S6.

### `relaxations/`
This folder is dedicated to data and scripts related to relaxation simulations, especially concerning domain structures and interlayer distances.

- **`relaxed_data`**: (directory) Contains all the LAMMPS data and xyz files of the relaxed twisted bilayers.
- **`rigid_data`**: (directory) Contains all the LAMMPS data files for the initial rigid structures of twisted bilayers.
- **`generate_rigid.py`**: Python script to make LAMMPS data files for the rigid structures of twisted bilayers and stores them in 'rigid_data' directory.
- **`mkrun_homo.sh`**: A shell script for setting up and running LAMMPS relaxation for a particular homobilayer system at a certain twist angle.
- **`mpirun_hetero.sh`**: A shell script for setting up and running LAMMPS relaxation for a particular heterobilayer system at a certain twist angle.
- **`utils.py`**: A Python script containing common utility functions used in the relaxations directory.
- **`plot_interlayer_distances.ipynb`**: Jupyter Notebook for plotting interpolating interlayer distances grids for the relaxed systems. The grids are stored in folder 'grid_data'.
Generates the combined heatmaps stored in heatmaps/combined/, which are presented in Figures 2, 4, and S5. Also contains a section that converts relaxed twirled structures of MoSe2/WSe2 to MoS2/WS2 to evade local minimum untwirled structure.
- **`interlayer_d_profiles.ipynb`**: Jupyter Notebook for plotting interlayer distance profiles between second nearest neighbour nodes. Stores the 1D interlayer distance grids in the 'grid_data_2nn' folder.
Generates the images 'interlayer_d_profile_MoS2.png' and 'interlayer_d_profile_MoS2.png' used in Figures 3 and S7. Also includes experimental comparison for MoS2 homobilayer where MLIP calculated
stacking area percentages of nodes, domain walls and domains are benchmarked against their experimental counterpart. This generates 'stacking_percents.png' used as Figure 7.
Also, generates the interlayer distance profiles with locations of node and domain wall centers and edges 'node_wall_area.png' used as Figure S8. 
- **`plot_domain_ratios.ipynb`**: A Jupyter Notebook for calculating domain ratios and storing them in 'domain_ratios.pkl' (with error bars).
Generates the images 'domain_ratio_schematic.png' used in Figure S11, and 'domain_ratio.png' used in Figure 6.
- **`plot_twirl.ipynb`**: Jupyter Notebook for calculating distortion angle of domain wall between nodes in heterobilayers, and stores them in 'twirl_angles.pkl'.
Generates the images 'twirl_schematic.png' used in Figure S10 and 'twirl_angles.png' used in Figure 5. 
